# 武汉理工大学 刘俊杰 2021
# https://www.zhihu.com/people/da-xia-tian-60
import numpy as np
import matplotlib.pyplot as plt
import models

model = models.exp2() # 设置算例模型

def gen_data(T):
	xs = np.zeros((T,model.nx))
	xs[0] = model.x0
	zs = np.zeros((T,model.nz))
	zs[0] = model.HP(xs[0],0)
	for t in range(1,T):
		xs[t] = model.FP(xs[t-1],t)
		zs[t] = model.HP(xs[t],t)
	return xs,zs

T = 300
M = 100
xss = []
zss = []
for i in range(M):
	xs,zs = gen_data(T)
	xss.append(xs)
	zss.append(zs)
	
	# 绘制3D图线
	# from mpl_toolkits.mplot3d import axes3d
	# fig = plt.figure()
	# ax = fig.gca(projection='3d')
	# ax.set_title("3D_Curve")
	# ax.set_xlabel("x")
	# ax.set_ylabel("y")
	# ax.set_zlabel("z")
	# figure = ax.plot(xs[:,0], xs[:,2], xs[:,4], c='r')

	# 绘制2D图线
	# plt.plot(xs[:,0], xs[:,2])
	# plt.show()
np.savez('data',xss = xss,zss = zss) # 将数据存为npz文件
