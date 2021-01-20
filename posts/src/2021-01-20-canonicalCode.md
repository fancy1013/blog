---
title: "Canonical Voting 代码详解"
data: Jan 20, 2021
---

# 前置知识

1.`__global__` 

This is a CUDA C keyword (declaration specifier)

核函数的声明。



2.`atomicAdd`

 int **atomicAdd**(int* address, int val); reads the 32-bit word old from the location pointed to by address in global or shared memory, computes (old + val), and stores the result back to memory at the same address. The function returns old.



3.用CUDA实现加法.

```cuda
__global__ void add(float* x, float* y, float* z, int n)
{
	int index = threadIdx.x + blockIdx.x*blockDim.x;//全局索引
	int stride = blockDim.x * gridDim.x;//步长
	for (int i=index;i<n;i+=stride)
	{
		z[i] = x[i]+y[i];
	}
}
```

main:

```cuda
int main()
{
	int N=1<<20;
	int nBytes = N * sizeof(float);
	
	float *x, *y, *z;
	x = (float*)malloc(nBytes);
	y = (float*)malloc(nBytes);
	z = (float*)malloc(nBytes);
	//这里是申请了存放向量的地址空间,并转化为float类型的指针
	
	for(int i=0;i<N;++i)
	{
		x[i] = 10.0;
		y[i] = 20.0;
	}
	//对向量的值进行初始化
	
	float *d_x,*d_y,*d_z;
	cudaMalloc((void**)&d_x, nBytes);//这个函数申请了在device上的内存
	cudaMalloc((void**)&d_y, nBytes);
	cudaMalloc((void**)&d_z, nBytes);
	//申请在device上的内存
	
	cudaMemcpy((void*)d_x, (void*)x, nBytes, cudaMemcpyHostToDevice);
	cudaMemcpy((void*)d_y, (void*)y, nBytes, cudaMemcpyHostToDevice);
	//将host数据拷贝到device上
	
	dim3 blockSize(256);//定义kernel执行的block数量
	dim3 gridSize((N+blockSize.x-1)/blockSize.x);
	
	add<<<gridSize, blockSize>>>(d_x,d_y,d_z,N);
	//执行kernel
	
	cudaMemcpy((void*)z,(void*)d_z,mBytes,cudaMemcpyHostToDevice);
	//将结果从divice拷贝到host
	
	cudaFree(d_x);
	cudaFree(d_y);
	cudaFree(d_z);
	//释放device内存
	
	free(x);
	free(y);
	free(z);
	//释放host内存
	
	return 0;
}
```



引入cudaMallocManaged函数对上面的代码优化：

```cuda
int main()
{
    int N = 1 << 20;
    int nBytes = N * sizeof(float);
    
    // 申请托管内存
    float *x, *y, *z;
    cudaMallocManaged((void**)&x, nBytes);
    cudaMallocManaged((void**)&y, nBytes);
    cudaMallocManaged((void**)&z, nBytes);

	// 初始化数据
    for (int i = 0; i < N; ++i)
    {
        x[i] = 10.0;
        y[i] = 20.0;
    }

    // 定义kernel的执行配置
    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);
    // 执行kernel
    add << < gridSize, blockSize >> >(x, y, z, N);

    // 同步device 保证结果能正确访问
    cudaDeviceSynchronize();
    
    // 检查执行结果
    float maxError = 0.0;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(z[i] - 30.0));
    std::cout << "最大误差: " << maxError << std::endl;

    // 释放内存
    cudaFree(x);
    cudaFree(y);
    cudaFree(z);

    return 0;
}

```



4.`__restrict__`

CUDA官方文档的解释：By making a, b, and c restricted pointers, the programmer asserts to the compiler that the pointers are in fact not aliased, which in this case means writes through c would never overwrite elements of a or b. 

大概说指针指向的内容不会被其他指针修改。

# 一、hv_cuda_kernel.cu

**这个文件是Canonical voting process实现的核心文件。**

**使用CUDA编写，进而对pytorch进行扩展。**

首先是包含的库。

```cuda
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include "helper_math.h"
#include <thrust/device_vector.h>

#include <vector>
#include <iostream>
```



接着是一个前向传播的核函数，和论文中的Algorithm1流程基本相同。

```cuda
template <typename scalar_t>
__global__ void hv_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> points,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> xyz_labels,//即LLC坐标
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> scale_labels,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> obj_labels,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> grid_obj,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grid_rot,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grid_scale,
    float3 corner,
    const float* __restrict__ res,
    const int* __restrict__ num_rots)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;//即线程的全局索引
    if (c < points.size(0)) {
        
        scalar_t objness = obj_labels[c];//即Alg1中的objectness
        float3 corr = make_float3(
            xyz_labels[c][0] * scale_labels[c][0],
            xyz_labels[c][1] * scale_labels[c][1],
            xyz_labels[c][2] * scale_labels[c][2]
        );//即Alg1中的v_i
        
        float3 point = make_float3(points[c][0], points[c][1], points[c][2]);//即point的坐标
        const float rot_interval = 2 * 3.141592654f / (*num_rots);//即旋转的角度的间隔，将360度分成K个。
        
        for (int i = 0; i < (*num_rots); i++) { //寻找可能的朝向，K次循环
            float theta = i * rot_interval;
            float3 offset = make_float3(-cos(theta) * corr.x + sin(theta) * corr.z,
                -corr.y, -sin(theta) * corr.x - cos(theta) * corr.z);//这个offset是什么呢？
            float3 center_grid = (point + offset - corner) / (*res);
            if (center_grid.x < 0 || center_grid.y < 0 || center_grid.z < 0 || 
                center_grid.x >= grid_obj.size(0) - 1 || center_grid.y >= grid_obj.size(1) - 1 || center_grid.z >= grid_obj.size(2) - 1) {
                continue;
            }
            int3 center_grid_floor = make_int3(center_grid);
            int3 center_grid_ceil = center_grid_floor + 1;
            float3 residual = fracf(center_grid);
            
            float3 w0 = 1.f - residual;
            float3 w1 = residual;
            
            //附近8个格点的投票
            float lll = w0.x * w0.y * w0.z * objness;
            float llh = w0.x * w0.y * w1.z * objness;
            float lhl = w0.x * w1.y * w0.z * objness;
            float lhh = w0.x * w1.y * w1.z * objness;
            float hll = w1.x * w0.y * w0.z * objness;
            float hlh = w1.x * w0.y * w1.z * objness;
            float hhl = w1.x * w1.y * w0.z * objness;
            float hhh = w1.x * w1.y * w1.z * objness;
            
			//用atomicAdd对投票结果进行累积
            atomicAdd(&grid_obj[center_grid_floor.x][center_grid_floor.y][center_grid_floor.z], lll);
            atomicAdd(&grid_obj[center_grid_floor.x][center_grid_floor.y][center_grid_ceil.z], llh);
            atomicAdd(&grid_obj[center_grid_floor.x][center_grid_ceil.y][center_grid_floor.z], lhl);
            atomicAdd(&grid_obj[center_grid_floor.x][center_grid_ceil.y][center_grid_ceil.z], lhh);
            atomicAdd(&grid_obj[center_grid_ceil.x][center_grid_floor.y][center_grid_floor.z], hll);
            atomicAdd(&grid_obj[center_grid_ceil.x][center_grid_floor.y][center_grid_ceil.z], hlh);
            atomicAdd(&grid_obj[center_grid_ceil.x][center_grid_ceil.y][center_grid_floor.z], hhl);
            atomicAdd(&grid_obj[center_grid_ceil.x][center_grid_ceil.y][center_grid_ceil.z], hhh);

            float rot_vec[2] = {cos(theta), sin(theta)};
            for (int j = 0; j < 2; j++) {
                float rot = rot_vec[j];
                atomicAdd(&grid_rot[center_grid_floor.x][center_grid_floor.y][center_grid_floor.z][j], lll * rot);
                atomicAdd(&grid_rot[center_grid_floor.x][center_grid_floor.y][center_grid_ceil.z][j], llh * rot);
                atomicAdd(&grid_rot[center_grid_floor.x][center_grid_ceil.y][center_grid_floor.z][j], lhl * rot);
                atomicAdd(&grid_rot[center_grid_floor.x][center_grid_ceil.y][center_grid_ceil.z][j], lhh * rot);
                atomicAdd(&grid_rot[center_grid_ceil.x][center_grid_floor.y][center_grid_floor.z][j], hll * rot);
                atomicAdd(&grid_rot[center_grid_ceil.x][center_grid_floor.y][center_grid_ceil.z][j], hlh * rot);
                atomicAdd(&grid_rot[center_grid_ceil.x][center_grid_ceil.y][center_grid_floor.z][j], hhl * rot);
                atomicAdd(&grid_rot[center_grid_ceil.x][center_grid_ceil.y][center_grid_ceil.z][j], hhh * rot);
            }

            for (int j = 0; j < 3; j++) {
                float scale = scale_labels[c][j];
                atomicAdd(&grid_scale[center_grid_floor.x][center_grid_floor.y][center_grid_floor.z][j], lll * scale);
                atomicAdd(&grid_scale[center_grid_floor.x][center_grid_floor.y][center_grid_ceil.z][j], llh * scale);
                atomicAdd(&grid_scale[center_grid_floor.x][center_grid_ceil.y][center_grid_floor.z][j], lhl * scale);
                atomicAdd(&grid_scale[center_grid_floor.x][center_grid_ceil.y][center_grid_ceil.z][j], lhh * scale);
                atomicAdd(&grid_scale[center_grid_ceil.x][center_grid_floor.y][center_grid_floor.z][j], hll * scale);
                atomicAdd(&grid_scale[center_grid_ceil.x][center_grid_floor.y][center_grid_ceil.z][j], hlh * scale);
                atomicAdd(&grid_scale[center_grid_ceil.x][center_grid_ceil.y][center_grid_floor.z][j], hhl * scale);
                atomicAdd(&grid_scale[center_grid_ceil.x][center_grid_ceil.y][center_grid_ceil.z][j], hhh * scale);
            }
            
        }
    }
}

```



