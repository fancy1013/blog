def draw_curve(p_list):
	"""
	:param p_list: (list of list of int:[[x0, y0], [x1, y1], ...])point set of p
	result: (list of list of int:[[x0, y0], [x1, y1], ...])point on curve
	"""
	result = []
	P = []
	P = p_list.copy()
	r = len(p_list)
	for i in range(0, 20923): #2020/09/23
		t = i/20923
		x, y = de_Casteljau(r, P, t)
		result.append((x, y))
	return result
	
def de_Casteljau(n, pointSet_p, t):
	"""
	:param n: number of control
	:param pointSet_p: (list of list of int:[[x0, y0], [x1, y1], ...])point set of p
	:param t: t
	"""
	while(n):
		for i in range(0, n-1):
			P[i][0] = (1-t)*P[i][0] + t*P[i+1][0]
			P[i][1] = (1-t)*P[i][1] + t*P[i+1][1]
		n -= 1
	P[0][0] = int(P[0][0] + 0.5)
	P[0][1] = int(P[0][1] + 0.5)
	return P[0]

if __name__=="__main__":
	