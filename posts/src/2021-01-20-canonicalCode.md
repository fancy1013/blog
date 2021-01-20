---
title: "Canonical Voting Algorithm: Code"
data: Jan 20, 2021
---

[TOC]

# 零、前置知识
[TOC]

1.`__global__` 

This is a CUDA C keyword (declaration specifier)

核函数的声明。



2.`atomicAdd`

 int **atomicAdd**(int* address, int val); reads the 32-bit word old from the location pointed to by address in global or shared memory, computes (old + val), and stores the result back to memory at the same address. The function returns old.



3.用CUDA实现加法.

```c
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

```c
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

```c
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
[TOC]

**这个文件是Canonical voting process实现的核心文件，主要实现了算法一。**

**使用CUDA编写，进而对pytorch进行扩展。**

## INCULDE

```c
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include "helper_math.h"
#include <thrust/device_vector.h>

#include <vector>
#include <iostream>
```



## hv_cuda_forward_kernel

一个前向传播的核函数，这个函数的流程**和论文中的Algorithm1流程基本相同。**

```c
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
            
			//用atomicAdd对投票结果进行累积，要对三个需要投票的进行累积，所以下面是三个循环
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



## hv_cuda_average_kernel

**对应Algorithm1中的14行：Normalize。**

```c
template <typename scalar_t>
__global__ void hv_cuda_average_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> grid,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grid_rot,
    torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> grid_scale)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= grid.size(0) || y >= grid.size(1) || z >= grid.size(2)) return;

    float w = grid[x][y][z];
    for (int j = 0; j < 2; j++) {
        grid_rot[x][y][z][j] /= w + 1e-7;
    }
    for (int j = 0; j < 3; j++) {
        grid_scale[x][y][z][j] /= w + 1e-7;
    }
}
```



## hv_cuda_forward
**对之前的hv_cuda_forward_kernel进行一个封装，返回投票结果。**


```c
std::vector<torch::Tensor> hv_cuda_forward(
    torch::Tensor points,
    torch::Tensor xyz_labels,
    torch::Tensor scale_labels,
    torch::Tensor obj_labels,
    torch::Tensor res,
    torch::Tensor num_rots) 
{
    auto corners = torch::stack({std::get<0>(torch::min(points, 0)), std::get<0>(torch::max(points, 0))}, 0);  // 2 x 3
    auto corner = corners[0];  // 3
    auto diff = (corners[1] - corners[0]) / res;  // 3
    auto grid_obj = torch::zeros({diff[0].item().to<int>() + 1, diff[1].item().to<int>() + 1, diff[2].item().to<int>() + 1}, points.options());
    auto grid_rot = torch::zeros({diff[0].item().to<int>() + 1, diff[1].item().to<int>() + 1, diff[2].item().to<int>() + 1, 2}, points.options());
    auto grid_scale = torch::zeros({diff[0].item().to<int>() + 1, diff[1].item().to<int>() + 1, diff[2].item().to<int>() + 1, 3}, points.options());
    
    // std::cout << grid.size(0) << ", " << grid.size(1) << ", " << grid.size(2) << std::endl;
    // std::cout << corner << std::endl;
    
    const int threads = 1024;
    const dim3 blocks((points.size(0) + threads - 1) / threads);
    
    AT_DISPATCH_FLOATING_TYPES(points.type(), "hv_forward_cuda", ([&] {
        hv_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
            points.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            xyz_labels.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            scale_labels.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            obj_labels.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            grid_obj.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            grid_rot.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            grid_scale.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            make_float3(corner[0].item().to<float>(), corner[1].item().to<float>(), corner[2].item().to<float>()),
            res.data<float>(),
            num_rots.data<int>()
        );
      }));
    
    AT_DISPATCH_FLOATING_TYPES(points.type(), "hv_average_cuda", ([&] {
        hv_cuda_average_kernel<scalar_t><<<dim3((grid_obj.size(0) + 7) / 8, (grid_obj.size(1) + 7) / 8, (grid_obj.size(2) + 7) / 8), dim3(8, 8, 8)>>>(
            grid_obj.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            grid_rot.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            grid_scale.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>()
        );
    }));
    return {grid_obj, grid_rot, grid_scale};
}
```



## hv_cuda_backward_kernel

实现反向传播的kernel部分。

前半部分和forward几乎一样。

疑问：后半部分是什么意思呢？

```c
template <typename scalar_t>
__global__ void hv_cuda_backward_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> grad_grid,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> points,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> xyz_labels,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> scale_labels,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> obj_labels,
    // torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> d_points,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> d_xyz_labels,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> d_scale_labels,
    torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> d_obj_labels,
    float3 corner,
    const float* __restrict__ res,
    const int* __restrict__ num_rots)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < points.size(0)) {
        
        scalar_t objness = obj_labels[c];
        float3 corr = make_float3(
            xyz_labels[c][0] * scale_labels[c][0],
            xyz_labels[c][1] * scale_labels[c][1],
            xyz_labels[c][2] * scale_labels[c][2]
        );
        float3 point = make_float3(points[c][0], points[c][1], points[c][2]);
        float rot_interval = 2 * 3.141592654f / (*num_rots);
        for (int i = 0; i < (*num_rots); i++) {
            float theta = i * rot_interval;
            float3 offset = make_float3(-cos(theta) * corr.x + sin(theta) * corr.z,
                -corr.y, -sin(theta) * corr.x - cos(theta) * corr.z);
            float3 center_grid = (point + offset - corner) / (*res);
            if (center_grid.x < 0 || center_grid.y < 0 || center_grid.z < 0 || 
                center_grid.x >= grad_grid.size(0) - 1 || center_grid.y >= grad_grid.size(1) - 1 || center_grid.z >= grad_grid.size(2) - 1) {
                continue;
            }
            int3 center_grid_floor = make_int3(center_grid);
            int3 center_grid_ceil = center_grid_floor + 1;
            float3 residual = fracf(center_grid);
            
            float3 w0 = 1.f - residual;
            float3 w1 = residual;
            
            d_obj_labels[c] += grad_grid[center_grid_floor.x][center_grid_floor.y][center_grid_floor.z] * w0.x * w0.y * w0.z;
            d_obj_labels[c] += grad_grid[center_grid_floor.x][center_grid_floor.y][center_grid_ceil.z] * w0.x * w0.y * w1.z;
            d_obj_labels[c] += grad_grid[center_grid_floor.x][center_grid_ceil.y][center_grid_floor.z] * w0.x * w1.y * w0.z;
            d_obj_labels[c] += grad_grid[center_grid_floor.x][center_grid_ceil.y][center_grid_ceil.z] * w0.x * w1.y * w1.z;
            d_obj_labels[c] += grad_grid[center_grid_ceil.x][center_grid_floor.y][center_grid_floor.z] * w1.x * w0.y * w0.z;
            d_obj_labels[c] += grad_grid[center_grid_ceil.x][center_grid_floor.y][center_grid_ceil.z] * w1.x * w0.y * w1.z;
            d_obj_labels[c] += grad_grid[center_grid_ceil.x][center_grid_ceil.y][center_grid_floor.z] * w1.x * w1.y * w0.z;
            d_obj_labels[c] += grad_grid[center_grid_ceil.x][center_grid_ceil.y][center_grid_ceil.z] * w1.x * w1.y * w1.z;

            float3 dgrid_dcenter = make_float3(
                - grad_grid[center_grid_floor.x][center_grid_floor.y][center_grid_floor.z] * w0.y * w0.z
                - grad_grid[center_grid_floor.x][center_grid_floor.y][center_grid_ceil.z] * w0.y * w1.z
                - grad_grid[center_grid_floor.x][center_grid_ceil.y][center_grid_floor.z] * w1.y * w0.z
                - grad_grid[center_grid_floor.x][center_grid_ceil.y][center_grid_ceil.z] * w1.y * w1.z
                + grad_grid[center_grid_ceil.x][center_grid_floor.y][center_grid_floor.z] * w0.y * w0.z
                + grad_grid[center_grid_ceil.x][center_grid_floor.y][center_grid_ceil.z] * w0.y * w1.z
                + grad_grid[center_grid_ceil.x][center_grid_ceil.y][center_grid_floor.z] * w1.y * w0.z
                + grad_grid[center_grid_ceil.x][center_grid_ceil.y][center_grid_ceil.z] * w1.y * w1.z,
                - grad_grid[center_grid_floor.x][center_grid_floor.y][center_grid_floor.z] * w0.x * w0.z
                - grad_grid[center_grid_floor.x][center_grid_floor.y][center_grid_ceil.z] * w0.x * w1.z
                + grad_grid[center_grid_floor.x][center_grid_ceil.y][center_grid_floor.z] * w0.x * w0.z
                + grad_grid[center_grid_floor.x][center_grid_ceil.y][center_grid_ceil.z] * w0.x * w1.z
                - grad_grid[center_grid_ceil.x][center_grid_floor.y][center_grid_floor.z] * w1.x * w0.z
                - grad_grid[center_grid_ceil.x][center_grid_floor.y][center_grid_ceil.z] * w1.x * w1.z
                + grad_grid[center_grid_ceil.x][center_grid_ceil.y][center_grid_floor.z] * w1.x * w0.z
                + grad_grid[center_grid_ceil.x][center_grid_ceil.y][center_grid_ceil.z] * w1.x * w1.z,
                - grad_grid[center_grid_floor.x][center_grid_floor.y][center_grid_floor.z] * w0.x * w0.y
                + grad_grid[center_grid_floor.x][center_grid_floor.y][center_grid_ceil.z] * w0.x * w0.y
                - grad_grid[center_grid_floor.x][center_grid_ceil.y][center_grid_floor.z] * w0.x * w1.y
                + grad_grid[center_grid_floor.x][center_grid_ceil.y][center_grid_ceil.z] * w0.x * w1.y
                - grad_grid[center_grid_ceil.x][center_grid_floor.y][center_grid_floor.z] * w1.x * w0.y
                + grad_grid[center_grid_ceil.x][center_grid_floor.y][center_grid_ceil.z] * w1.x * w0.y
                - grad_grid[center_grid_ceil.x][center_grid_ceil.y][center_grid_floor.z] * w1.x * w1.y
                + grad_grid[center_grid_ceil.x][center_grid_ceil.y][center_grid_ceil.z] * w1.x * w1.y) * objness;
            
            // d_points[c][0] += dgrid_dcenter.x;
            // d_points[c][1] += dgrid_dcenter.y;
            // d_points[c][2] += dgrid_dcenter.z;

            float3 d_corr = make_float3(- cos(theta) * dgrid_dcenter.x - sin(theta) * dgrid_dcenter.z,
                -dgrid_dcenter.y, sin(theta) * dgrid_dcenter.x - cos(theta) * dgrid_dcenter.z);

            d_xyz_labels[c][0] += d_corr.x * scale_labels[c][0];
            d_xyz_labels[c][1] += d_corr.y * scale_labels[c][1];
            d_xyz_labels[c][2] += d_corr.z * scale_labels[c][2];

            d_scale_labels[c][0] += d_corr.x * xyz_labels[c][0];
            d_scale_labels[c][1] += d_corr.y * xyz_labels[c][1];
            d_scale_labels[c][2] += d_corr.z * xyz_labels[c][2];
        }
    }
}


```



接着是一个叫hv_cuda_backward的函数，封装了kernel。

```c
std::vector<torch::Tensor> hv_cuda_backward(
    torch::Tensor grad_grid,
    torch::Tensor points,
    torch::Tensor xyz_labels,
    torch::Tensor scale_labels,
    torch::Tensor obj_labels,
    torch::Tensor res,
    torch::Tensor num_rots) 
{
    auto corners = torch::stack({std::get<0>(torch::min(points, 0)), std::get<0>(torch::max(points, 0))}, 0);  // 2 x 3
    auto corner = corners[0];  // 3
    auto diff = (corners[1] - corners[0]) / res;  // 3
    // auto d_points = torch::zeros_like(points);
    auto d_xyz_labels = torch::zeros_like(xyz_labels);
    auto d_scale_labels = torch::zeros_like(scale_labels);
    auto d_obj_labels = torch::zeros_like(obj_labels);
    
    const int threads = 512;
    const dim3 blocks((points.size(0) + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES(points.type(), "hv_backward_cuda", ([&] {
        hv_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
            grad_grid.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            points.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            xyz_labels.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            scale_labels.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            obj_labels.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            // d_points.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            d_xyz_labels.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            d_scale_labels.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            d_obj_labels.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            make_float3(corner[0].item().to<float>(), corner[1].item().to<float>(), corner[2].item().to<float>()),
            res.data<float>(),
            num_rots.data<int>()
        );
      }));
    return {d_xyz_labels, d_scale_labels, d_obj_labels};
}
```



接着实现了一个2维版本的canonical hough voting，暂且不做记录。



# 二、hv_cuda.cpp

[TOC]

主要是对hv_cuda_kernel.cu的内容进行封装。

定义了一些函数来检查输入。

```c++
#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
```



首先，声明CUDA的函数：

```c++
std::vector<torch::Tensor> hv_cuda_forward(
    torch::Tensor points,
    torch::Tensor xyz_labels,
    torch::Tensor scale_labels,
    torch::Tensor obj_labels,
    torch::Tensor res,
    torch::Tensor num_rots);
```



然后，定义torch可以调用的函数：

```c++
std::vector<torch::Tensor> hv_forward(
    torch::Tensor points,
    torch::Tensor xyz_labels,
    torch::Tensor scale_labels,
    torch::Tensor obj_labels,
    torch::Tensor res,
    torch::Tensor num_rots) {
  CHECK_INPUT(points);
  CHECK_INPUT(xyz_labels);
  CHECK_INPUT(scale_labels);
  CHECK_INPUT(obj_labels);
  CHECK_INPUT(res);
  CHECK_INPUT(num_rots);

  return hv_cuda_forward(points, xyz_labels, scale_labels, obj_labels, res, num_rots);
}
```



用类似的流程，定义了hv_backward,hv2d_forward。



并且，在最后用PYBIND11进行链接。

```c++
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &hv_forward, "hv forward (CUDA)");
  m.def("backward", &hv_backward, "hv backward (CUDA)");
  m.def("forward2d", &hv2d_forward, "hv backward (CUDA)");
  // m.def("2dbackward", &hv2d_backward, "hv backward (CUDA)");
}
```



# 三、setup.py

用来进行模块的安装。

```python
from platform import version
from setuptools import setup 
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='houghvoting', #总的模块的名字
    version='0.0.1',#版本
    ext_modules=[
        CUDAExtension('houghvoting.cuda', [ # 用CUDA extention扩展出来的模块的名字
            'src/hv_cuda.cpp',             #链接的文件
            'src/hv_cuda_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
```

这里有两个名字：houghvoting和houghvoting.cuda，前者是外部调用时的库名，后者是这个库內部构建时调用的库名。



# 四、voting.py

