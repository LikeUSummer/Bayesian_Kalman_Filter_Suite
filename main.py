# 武汉理工大学 刘俊杰 2021
# https://www.zhihu.com/people/da-xia-tian-60
import numpy as np
import numdifftools as nd
from numpy.core.defchararray import mod
import scipy.stats as st
import matplotlib.pyplot as plt
from scipy.stats.stats import mode
import time
import models
import filters

"""
配置仿真核心参数
"""
model = models.exp2() #算例模型
filters.init(model) #根据模型参数初始化部分滤波器的全局参数
T = 100 #仿真步数
M = 10 #仿真次数
data = None #可从文件加载仿真数据，便于对比或加速仿真，如 np.load('test.pyc')，若设置为None，则会动态生成数据
error_mask = np.array([1]) #标记估计状态中需要统计误差的维度，如 np.array([1, 1, 1, 0, 0, 0])

"""
配置滤波器
解开相应注释以激活需要仿真对比的滤波器
"""
filters_used = {}
filters_used['UKF'] = filters.UKF
filters_used['IUKF'] = filters.IUKF
filters_used['CKF'] = filters.CKF
# filters_used['CKF_SR'] = filters.CKF_SR
filters_used['ICKF'] = filters.ICKF
# filters_used['ICKF_SR'] = filters.ICKF_SR
# filters_used['ICKF_D'] = filters.ICKF_D

# filters_used['HUKF'] = filters.HUKF
# filters_used['ALA_RUKF'] = filters.ALA_RUKF
# filters_used['ALA_RIUKF'] = filters.ALA_RIUKF
filters_used['HCKF'] = filters.HCKF
# filters_used['HCIF'] = filters.HCIF
# filters_used['IGG-HCKF'] = filters.IGG_CKF
filters_used['ALA-RCKF'] = filters.ALA_RCKF
filters_used['ALA-RICKF'] = filters.ALA_RICKF
# filters_used['L-RICKF'] = filters.L_RICKF
# filters_used['ALA_RICKF_SR'] = filters.ALA_RICKF_SR
# filters_used['HICKF'] = filters.HICKF

# filters_used['PF'] = filters.PF
# filters_used['UPF'] = filters.KPF(filters.UKF).filter
# filters_used['IUPF'] = filters.KPF(filters.IUKF).filter
# filters_used['CPF'] = filters.KPF(filters.CKF).filter
# filters_used['CPF_SR'] = filters.KPF(filters.CKF_SR).filter
# filters_used['ICPF'] = filters.KPF(filters.ICKF).filter
# filters_used['ICPF_SR'] = filters.KPF(filters.ICKF_SR).filter

# filters_used['HCPF'] = filters.KPF(filters.HCKF).filter
# filters_used['ALA_RCPF'] = filters.KPF(filters.ALA_RCKF).filter
# filters_used['ALA_RICPF'] = filters.KPF(filters.ALA_RICKF).filter
# filters_used['ALA_RUPF'] = filters.KPF(filters.ALA_RUKF).filter
# filters_used['ALA_RIUPF'] = filters.KPF(filters.ALA_RIUKF).filter

'''
仿真计算
'''
# 高斯滤波器加载器
def GF_simulator(xs, zs, filter, x0):
    P_xx = model.P0
    xes = xs.copy()
    t1 = time.time()
    for t in range(1, T):
        if t == 1: # 状态初值，可设置固定值，也可根据初始协方差动态生成
            x = x0
        else:
            x = xes[t - 1]
        r = filter(model, x, zs[t], P_xx, t)
        xes[t] = r[0]
        P_xx = r[1]
    t2 = time.time()
    return xes, (xes - xs) ** 2, t2 - t1

# 粒子滤波器加载器
def PF_simulator(xs, zs, filter, x0):
    N = 200
    ps = np.array([x0] * N)
    xes = xs.copy()
    P_xx = model.P0
    t1 = time.time()
    for t in range(1, T):
        xes[t], P_xx, ps = filter(model, ps, zs[t], P_xx, t)
    t2 = time.time()
    return xes, (xes - xs) ** 2, t2 - t1

# 用于动态生成仿真数据
def gen_data(T):
    xs = np.zeros((T, model.nx))
    xs[0] = model.x0
    zs = np.zeros((T, model.nz))
    zs[0] = model.HP(xs[0], 0)
    for t in range(1, T):
        xs[t] = model.FP(xs[t - 1], t)
        zs[t] = model.HP(xs[t], t)
    return xs, zs

# 开始仿真
# paras = [10, 25, 50, 75, 100, 250, 500, 750, 1000]
# for p in paras:  # 批量仿真
# 	print('p =', p)
# 	model.kappa = p
errs = {}
xess = {}
times = {}
xs = []
zs = []
ts = range(T)

for name in filters_used:
    errs[name] = np.zeros((T, model.nx))
    times[name] = 0

for i in range(M):
    print('iteration:' + str(i))
    if data:
        xs = data['xss'][i]
        zs = data['zss'][i]
    else:
        xs, zs = gen_data(T) # 动态生成模拟数据
    x0 = np.random.multivariate_normal(xs[0], model.P0)
    for name, filter in filters_used.items():
        if name.find('PF') == -1:
            xes, err, dt = GF_simulator(xs, zs, filter, x0)
        else:
            xes, err, dt = PF_simulator(xs, zs, filter, x0)
        errs[name] += err
        xess[name] = xes
        times[name] += dt

# 仿真完成，计算平均误差
for name in filters_used:
    errs[name] = errs[name]@error_mask / M # 计算MSE，如果要统计全部维度，则直接使用np.sum(errs[name], axis=1)/M 
    errs[name] = np.sqrt(errs[name]) # 计算RMSE
    times[name] = times[name] / M / T
    print(name, np.mean(errs[name]), times[name]) # 输出平均误差和耗时

"""
绘制仿真结果
"""
# 配置绘图参数
linestyles = {}
markers = ['.', 'x', '+', '*']
i = 0
for name in filters_used:
    linestyles[name] = markers[i] + '-.'
    i = (i + 1) % 4
linestyles['ALA_RICKF'] = 'b^-' # 凸显样式

# 状态分量-时间
plt.figure(figsize = (8, 6), dpi = 150)
plt.grid(linestyle = "--")
plt.xlabel(u'time')
plt.ylabel(u'x', rotation = 90)
[plt.plot(xess[name], linestyles[name], linewidth = 1) for name in filters_used]
plt.plot(xs, 'r', linewidth = 1)
legends = [name for name in filters_used]
legends.append('True')
plt.legend(legends)

# 平均误差-时间
plt.figure(figsize = (8, 6), dpi = 150)
plt.xlabel(u'time')
plt.ylabel(u'err', rotation = 90)
[plt.plot(errs[name], linestyles[name], linewidth = 1) for name in filters_used]
plt.legend([name for name in filters_used])

# 二维相轨迹
# plt.figure(figsize = (8, 6), dpi = 150)
# [plt.plot(xess[name][1:, 0], xess[name][1:, 2], linestyles[name], linewidth = 1) for name in filters_used]
# plt.plot(xs[1:, 0], xs[1:, 2], 'r', linewidth = 1)
# plt.legend(legends)

plt.show()
