# 武汉理工大学 刘俊杰 2021
# https://www.zhihu.com/people/da-xia-tian-60
'''
实现了投影统计的相关工具函数及测试
'''
import numpy as np

def projection_statistics(points): #每个点按行向量排列
	axis = np.array([[1,0],[0,1]]) #坐标方向
	PS = np.zeros(len(points)) #记录各样本点PS值
	for u in axis:
		hu = (points@u).T
		med_hu = np.median(hu)
		r = np.median(np.abs(hu-med_hu))
		PS1 = np.abs(hu-med_hu)/(1.4826*r)
		index = np.where(PS1>PS)
		PS[index] = PS1[index]
	return PS

def adjust_weights(PS): #进行卡方检验，计算离群值的权值
	n = len(PS)
	X_2_0975 = 0.051
	weights = np.ones(n)
	for i in range(n):
		X_2 = PS[i]**2
		w = 2.25/X_2
		if X_2>X_2_0975 and w<1:
			weights[i] = w
	return weights

if __name__=='__main__':
	import matplotlib.pyplot as plt
	pts = np.random.multivariate_normal(np.ones(2),0.1*np.eye(2),10)	#主体分布
	outs = np.random.multivariate_normal(4*np.ones(2),0.1*np.eye(2),3)	#离群点
	pts = np.vstack([pts,outs])

	PS = projection_statistics(pts)
	ws = adjust_weights(PS)
	print(ws)

	plt.scatter(pts[:,0],pts[:,1])
	plt.show()
