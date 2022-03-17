# 武汉理工大学 刘俊杰 2021
# https://www.zhihu.com/people/da-xia-tian-60
'''
本文件是各种滤波器的实现，部分滤波器（如UKF）在使用前需要先调用init函数初始化相关参数（由算例模型决定）
滤波器命名规则：增加前缀表示改善了原滤波器性能，后缀表示原滤波器的一种(近似)等效构造
'''
import numpy as np
import numdifftools as nd
import scipy.linalg as sl
from numpy.core.defchararray import mod
import scipy.stats as st
from scipy.stats.stats import mode
from sympy import factor
import rs

lamda = None
wms = None
wcs = None
def init(model): # 为部分算法初始化全局参数，这些参数可能依赖于模型参数
	global lamda, wms, wcs
	alpha = 1
	beta = 2
	kappa = 3 - model.nx
	lamda = alpha ** 2 * (model.nx + kappa) - model.nx
	wms = np.block([lamda, 0.5 * np.ones(2 * model.nx)]) / (model.nx + lamda)
	wcs = wms.copy()
	wcs[0] += (1 - alpha ** 2 + beta)

# 无迹卡尔曼滤波器
def UKF(model, x, z, P_xx, t):
	S_xx = np.linalg.cholesky((lamda + model.nx) * P_xx)
	xs = np.block([np.zeros((model.nx, 1)), S_xx, -S_xx]).T + x	
	xps = np.block([[model.F(e, t)] for e in xs])
	xp = wms @ xps
	P_xx = (xps - xp).T @ np.diag(wcs) @ (xps - xp) + model.G @ model.Q @ model.G.T

	S_xx = np.linalg.cholesky((lamda + model.nx) * P_xx)
	xps = np.block([np.zeros((model.nx, 1)), S_xx, -S_xx]).T + xp
	zps = np.block([[model.H(e, t)] for e in xps])
	zp = wms @ zps
	P_zz = (zps - zp).T @ np.diag(wcs) @ (zps - zp) + model.R
	P_xz = (xps - xp).T @ np.diag(wcs) @ (zps - zp)
	K = P_xz @ np.linalg.inv(P_zz)
	xe = xp + K @ (z - zp).T
	P_xx = P_xx - K @ P_zz @ K.T
	
	return xe, P_xx

# 迭代无迹卡尔曼滤波器
def IUKF(model, x, z, P_xx, t):
	S_xx = np.linalg.cholesky((lamda + model.nx) * P_xx)
	xs = np.block([np.zeros((model.nx, 1)), S_xx, -S_xx]).T + x
	xps = np.block([[model.F(e, t)] for e in xs])
	xp = wms @ xps
	P_xx = (xps - xp).T @ np.diag(wcs) @ (xps - xp) + model.G @ model.Q @ model.G.T

	xe = xp
	dz0 = (z - model.H(xe, t)) * 1e3 # 乘一个大数，保证至少迭代两次，但不是必要的
	g = 1 # 卡尔曼增益衰减因子
	niter = 1
	if t > 0: # 初始迭代抑制策略
	 	niter = model.niter
	for i in range(niter):
		S_xx = np.linalg.cholesky((lamda + model.nx) * P_xx)
		xps = np.block([np.zeros((model.nx, 1)), S_xx, -S_xx]).T + xe
		zps = np.block([[model.H(e, t)] for e in xps])
		zp = wms @ zps
		P_zz = (zps - zp).T @ np.diag(wcs) @ (zps - zp) + model.R
		P_xz = (xps - xe).T @ np.diag(wcs) @ (zps - zp)
		K = P_xz @ np.linalg.inv(P_zz)
		xe = xe + g * K @ (z - zp) # 迭代更新状态估计值
		P_xx0 = P_xx # 暂存上一时刻的状态估计误差协方差
		P_xx = P_xx - g * K @ P_zz @ K.T # 更新当前时刻状态估计误差协方差
		g = model.eta * g # 卡尔曼增益衰减因子指数式减小
		# 迭代流程的似然面控制
		dx = xe - xp
		dz = z - model.H(xe, t)
		if dx @ np.linalg.inv(P_xx0) @ dx + dz @ model.Ri @ dz >= dz0 @ model.Ri @ dz0:
			break
		xp = xe # 在循环中xp表示上一次迭代的状态估计值
		dz0 = dz
	return xe, P_xx

# 基于Huber - M估计的无迹卡尔曼滤波器
def HUKF(model, x, z, P_xx, t):
	S_xx = np.linalg.cholesky((lamda + model.nx) * P_xx)
	xs = np.block([np.zeros((model.nx, 1)), S_xx, -S_xx]).T + x	
	xps = np.block([[model.F(e, t)] for e in xs])
	xp = wms @ xps
	P_xx = (xps - xp).T @ np.diag(wcs) @ (xps - xp) + model.Q

	S_xx = np.linalg.cholesky((lamda + model.nx) * P_xx)
	xps = np.block([np.zeros((model.nx, 1)), S_xx, -S_xx]).T + xp
	zps = np.block([[model.H(e, t)] for e in xps])
	zp = wms @ zps
	
	inn = z - zp # 新息
	e = np.abs(model.Rsi @ inn)
	km = 1.345
	phi = np.diag([i / km if i > km else 1 for i in e]) # Huber权矩阵
	R = model.Rs @ phi @ model.Rs.T
	P_zz = (zps - zp).T @ np.diag(wcs) @ (zps - zp) + R

	P_xz = (xps - xp).T @ np.diag(wcs) @ (zps - zp)
	K = P_xz @ np.linalg.inv(P_zz)
	xe = xp + K @ inn.T
	P_xx = P_xx - K @ P_zz @ K.T
	return xe, P_xx

# 基于近似最小一乘估计的无迹卡尔曼滤波器
def ALA_RUKF(model, x, z, P_xx, t):
	S_xx = np.linalg.cholesky((lamda + model.nx) * P_xx)
	xs = np.block([np.zeros((model.nx, 1)), S_xx, -S_xx]).T + x	
	xps = np.block([[model.F(e, t)] for e in xs])
	xp = wms @ xps
	P_xx = (xps - xp).T @ np.diag(wcs) @ (xps - xp) + model.G @ model.Q @ model.G.T
	S_xx = np.linalg.cholesky((lamda + model.nx) * P_xx)
	xps = np.block([np.zeros((model.nx, 1)), S_xx, -S_xx]).T + xp
	zps = np.block([[model.H(e, t)] for e in xps])
	zp = wms @ zps

	inn = z - zp
	phi = np.sqrt(inn @ model.Ri @ inn.T)
	km = 1.345
	factor = phi if phi > km else 1
	R = factor * model.R
	P_zz = (zps - zp).T @ np.diag(wcs) @ (zps - zp) + R

	P_xz = (xps - xp).T @ np.diag(wcs) @ (zps - zp)
	K = P_xz @ np.linalg.inv(P_zz)
	xe = xp + K @ inn.T
	P_xx = P_xx - K @ P_zz @ K.T
	return xe, P_xx

# 基于近似最小一乘估计的迭代无迹卡尔曼滤波器
def ALA_RIUKF(model, x, z, P_xx, t):
	S_xx = np.linalg.cholesky((lamda + model.nx) * P_xx)
	xs = np.block([np.zeros((model.nx, 1)), S_xx, -S_xx]).T + x
	xps = np.block([[model.F(e, t)] for e in xs])
	xp = wms @ xps
	P_xx = (xps - xp).T @ np.diag(wcs) @ (xps - xp) + model.G @ model.Q @ model.G.T

	xe = xp
	g = 1
	for i in range(model.niter):
		S_xx = np.linalg.cholesky((lamda + model.nx) * P_xx)
		xps = np.block([np.zeros((model.nx, 1)), S_xx, -S_xx]).T + xe
		zps = np.block([[model.H(e, t)] for e in xps])
		zp = wms @ zps

		inn = z - zp
		phi = np.sqrt(inn @ model.Ri @ inn.T)
		km = 1.345
		factor = phi if phi > km else 1
		R = factor * model.R
		P_zz = (zps - zp).T @ np.diag(wcs) @ (zps - zp) + R

		P_xz = (xps - xe).T @ np.diag(wcs) @ (zps - zp)
		K = P_xz @ np.linalg.inv(P_zz)
		xe = xe + g * K @ inn.T # P_xz.T @ np.linalg.inv(P_xx) @ (xp - xe))
		g = model.eta * g
		P_xx = P_xx - K @ P_zz @ K.T

		if np.linalg.norm(xp - xe) < model.iter_limit:
			break
		xp = xe
	return xe, P_xx

# 容积卡尔曼滤波器
def CKF(model, x, z, P_xx, t):
	m = 2 * model.nx
	S_xx = np.linalg.cholesky(model.nx * P_xx)
	xs = np.block([-S_xx, S_xx]).T + x
	xps = np.block([[model.F(e, t)] for e in xs])
	xp = np.mean(xps, axis = 0)
	P_xx = xps.T @ xps / m - np.outer(xp, xp) + model.G @ model.Q @ model.G.T

	S_xx = np.linalg.cholesky(model.nx * P_xx)
	xps = np.block([-S_xx, S_xx]).T + xp
	zps = np.block([[model.H(e, t)] for e in xps])
	zp = np.mean(zps, axis = 0)
	P_zz = zps.T @ zps / m - np.outer(zp, zp) + model.R
	P_xz = xps.T @ zps / m - np.outer(xp, zp)
	K = P_xz @ np.linalg.inv(P_zz)
	xe = xp + K @ (z - zp).T
	P_xx = P_xx - K @ P_zz @ K.T

	return xe, P_xx

# 平方根容积卡尔曼滤波器
def Tria(A):
	q, r = np.linalg.qr(A.T)
	return r.T
def CKF_SR(model, x, z, P_xx, t):
	m = 2 * model.nx
	S_xx = np.linalg.cholesky(P_xx)
	xs = np.block([-S_xx, S_xx]).T * np.sqrt(model.nx) + x
	xps = np.block([[model.F(e, t)] for e in xs])
	xp = np.mean(xps, axis = 0)
	Xs = (xps - xp).T / np.sqrt(m)
	Sp_xx = Tria(np.block([Xs, model.Qs]))

	xps = np.block([-Sp_xx, Sp_xx]).T * np.sqrt(model.nx) + xp
	zps = np.block([[model.H(e, t)] for e in xps])
	zp = np.mean(zps, axis = 0)
	Xps = (xps - xp).T / np.sqrt(m)
	Zps = (zps - zp).T / np.sqrt(m)
	S_zz = Tria(np.block([Zps, model.Rs]))
	P_xz = Xps @ Zps.T
	K = P_xz / S_zz.T / S_zz
	xe = xp + K @ (z - zp).T
	S_xx = Tria(np.block([Xps - K @ Zps, K @ model.Rs]))
	return xe, S_xx @ S_xx.T

# 迭代容积卡尔曼滤波
def ICKF(model, x, z, P_xx, t):  
	m = 2 * model.nx
	S_xx = np.linalg.cholesky(model.nx * P_xx)
	xs = np.block([-S_xx, S_xx]).T + x
	xps = np.block([[model.F(e, t)] for e in xs])
	xp = np.mean(xps, axis = 0)
	P_xx = xps.T @ xps / m - np.outer(xp, xp) + model.G @ model.Q @ model.G.T

	xe = xp
	dz0 = (z - model.H(xe, t)) * 1e3
	g = 1
	niter = 1
	if t > model.iter_start: # 初始迭代抑制策略
	 	niter = model.niter
	for i in range(niter):
		S_xx = np.linalg.cholesky(model.nx * P_xx)
		xps = np.block([-S_xx, S_xx]).T + xp
		xp = np.mean(xps, axis = 0)
		zps = np.block([[model.H(e, t)] for e in xps])
		zp = np.mean(zps, axis = 0)
		P_zz = zps.T @ zps / m - np.outer(zp, zp) + model.R
		P_xz = xps.T @ zps / m - np.outer(xp, zp)

		K = P_xz @ np.linalg.inv(P_zz)
		xe = xp + g * K @ (z - zp)
		g = model.eta * g
		P_xx0 = P_xx
		P_xx = P_xx - K @ P_zz @ K.T

		dx = xe - xp
		dz = z - model.H(xe, t)
		if dx @ np.linalg.inv(P_xx0) @ dx + dz @ model.Ri @ dz >= dz0 @ model.Ri @ dz0:
			break
		dz0 = dz
		xp = xe
	return xe, P_xx

# 基于求导的迭代容积卡尔曼滤波器
def ICKF_D(model, x, z, P_xx, t):  
	m = 2 * model.nx
	S_xx = np.linalg.cholesky(model.nx * P_xx)
	xs = np.block([-S_xx, S_xx]).T + x
	xps = np.block([[model.F(e, t)] for e in xs])
	xp = np.mean(xps, axis = 0)
	P_xx = xps.T @ xps / m - np.outer(xp, xp) + model.Q

	xe = xp
	for i in range(model.niter):
		H = model.JH(xe, t) # 量测函数Jacobian
		P_zz = H @ P_xx @ H.T + model.R
		P_xz = P_xx @ H.T
		K = P_xz @ np.linalg.inv(P_zz)
		xe = xe + K @ (z - model.H(xe, t)) # -P_xz.T @ np.linalg.inv(P_xx) @ (xp - xe))
		P_xx = P_xx - K @ P_zz @ K.T
		if np.linalg.norm(xp - xe) < model.iter_limit:
			break
		xp = xe
	return xe, P_xx

# 迭代平方根容积滤波器
def ICKF_SR(model, x, z, P_xx, t):
	m = 2 * model.nx
	S_xx = np.linalg.cholesky(P_xx)
	xs = np.block([-S_xx, S_xx]).T * np.sqrt(model.nx) + x
	xps = np.block([[model.F(e, t)] for e in xs])
	xp = np.mean(xps, axis = 0)
	Xs = (xps - xp).T / np.sqrt(m)
	S_xx = Tria(np.block([Xs, model.Qs]))

	xe = xp
	g = 1
	dz0 = (z - model.H(xe, t)) * 1e3
	niter = 1
	if t > model.iter_start: # 初始迭代抑制策略
	 	niter = model.niter
	for i in range(niter):
		xps = np.block([-S_xx, S_xx]).T * np.sqrt(model.nx) + xe
		zps = np.block([[model.H(e, t)] for e in xps])
		zp = np.mean(zps, axis = 0)
		Xps = (xps - xe).T / np.sqrt(m)
		Zps = (zps - zp).T / np.sqrt(m)
		S_zz = Tria(np.block([Zps, model.Rs]))
		P_xz = Xps @ Zps.T
		K = P_xz / S_zz.T / S_zz
		xe = xe + g * K @ (z - zp)
		g = g * model.eta
		P_xx0 = S_xx @ S_xx.T
		S_xx = Tria(np.block([Xps - K @ Zps, K @ model.Rs]))

		dx = xe - xp
		dz = z - model.H(xe, t)
		if dx @ np.linalg.inv(P_xx0) @ dx + dz @ model.Ri @ dz >= dz0 @ model.Ri @ dz0:
			break
		xp = xe
		dz0 = dz
	return xe, S_xx @ S_xx.T

# 基于Huber - M估计的容积卡尔曼滤波器
def HCKF(model, x, z, P_xx, t):
	n = 2 * model.nx
	S_xx = np.linalg.cholesky(model.nx * P_xx)
	xs = np.block([-S_xx, S_xx]).T + x
	xps = np.block([[model.F(e, t)] for e in xs])
	xp = np.mean(xps, axis = 0)
	P_xx = xps.T @ xps / n - np.outer(xp, xp)  + model.G @ model.Q @ model.G.T
	S_xx = np.linalg.cholesky(model.nx * P_xx)
	xps = np.block([-S_xx, S_xx]).T + xp
	zps = np.block([[model.H(e, t)] for e in xps])
	zp = np.mean(zps, axis = 0)

	inn = z - zp
	zr = np.abs(model.Rsi @ inn)
	km = 1.345
	phi = np.diag([e / km if e > km else 1 for e in zr])
	R = model.Rs @ phi @ model.Rs.T
	P_zz = zps.T @ zps / n - np.outer(zp, zp) + R

	P_xz = xps.T @ zps / n - np.outer(xp, zp)
	K = P_xz @ np.linalg.inv(P_zz)
	xe = xp + K @ inn.T
	P_xx = P_xx - K @ P_zz @ K.T

	return xe, P_xx

# 基于Huber - M估计的容积信息滤波器
def HCIF(model, x, z, P_xx, t):
	m = 2 * model.nx
	S_xx = np.linalg.cholesky(model.nx * P_xx)
	xs = np.block([-S_xx, S_xx]).T + x
	xps = np.block([[model.F(e, t)] for e in xs])
	xp = np.mean(xps, axis = 0)
	P_xx = xps.T @ xps / m - np.outer(xp, xp) + model.Q
	Y = np.linalg.inv(P_xx)
	y = Y @ xp

	S_xx = np.linalg.cholesky(model.nx * P_xx)
	xps = np.block([-S_xx, S_xx]).T + xp
	zps = np.block([[model.H(e, t)] for e in xps])
	zp = np.mean(zps, axis = 0)
	P_zz = zps.T @ zps / m - np.outer(zp, zp) + model.R
	P_xz = xps.T @ zps / m - np.outer(xp, zp)

	H = P_xz.T @ Y
	E = P_zz - H @ P_xx @ H.T
	Esi = np.linalg.inv(np.linalg.cholesky(E))
	inn = z - zp
	zr = np.abs(Esi @ inn)
	km = 1.345
	W = np.diag([km / e if e > km else 1 for e in zr])
	S = Esi.T @ W @ Esi

	y = y + H.T @ S @ (H @ xp + inn)
	Y = Y + H.T @ S @ H
	P_xx = np.linalg.inv(Y)
	xe = P_xx @ y
	return xe, P_xx

# 基于IGG分段权函数的容积卡尔曼滤波器
def IGG_CKF(model, x, z, P_xx, t):
	n = 2 * model.nx
	S_xx = np.linalg.cholesky(model.nx * P_xx)
	xs = np.block([-S_xx, S_xx]).T + x
	
	xps = np.block([[model.F(e, t)] for e in xs])
	xp = np.mean(xps, axis = 0)

	P_xx = xps.T @ xps / n - np.outer(xp, xp) + model.Q
	S_xx = np.linalg.cholesky(model.nx * P_xx)
	xps = np.block([-S_xx, S_xx]).T + xp
	zps = np.block([[model.H(e, t)] for e in xps])
	zp = np.mean(zps, axis = 0)

	inn = z - zp
	zr = np.abs(model.Rsi @ inn)
	km = 1.345
	phi = np.diag([e / km if e > km else 1 for e in zr])
	R = model.Rs @ phi @ model.Rs.T
	for e in zr:
		if e > 7:
			return xp, P_xx
	P_zz = zps.T @ zps / n - np.outer(zp, zp) + R

	P_xz = xps.T @ zps / n - np.outer(xp, zp)
	K = P_xz @ np.linalg.inv(P_zz)
	xe = xp + K @ inn
	P_xx = P_xx - K @ P_zz @ K.T
	return xe, P_xx

# 基于Huber - M估计的迭代容积卡尔曼滤波器
def HICKF(model, x, z, P_xx, t):
	m = 2 * model.nx
	S_xx = np.linalg.cholesky(model.nx * P_xx)
	xs = np.block([-S_xx, S_xx]).T + x
	xps = np.block([[model.F(e, t)] for e in xs])
	xp = np.mean(xps, axis = 0)
	P_xx = xps.T @ xps / m - np.outer(xp, xp) + model.Q

	xe = xp
	niter = 1
	if t > model.iter_start: # 初始迭代抑制策略
	 	niter = model.niter
	for i in range(niter):
		S_xx = np.linalg.cholesky(model.nx * P_xx)
		xps = np.block([-S_xx, S_xx]).T + xe
		zps = np.block([[model.H(e, t)] for e in xps])
		zp = np.mean(zps, axis = 0)

		inn = z - zp
		e = np.abs(model.Rsi @ inn)
		km = 1.345
		phi = np.diag([i / km if i > km else 1 for i in e])
		R = model.Rs @ phi @ model.Rs.T
		P_zz = zps.T @ zps / m - np.outer(zp, zp) + R

		P_xz = xps.T @ zps / m - np.outer(xp, zp)
		K = P_xz @ np.linalg.inv(P_zz)
		xe = xe + K @ inn.T
		P_xx = P_xx - K @ P_zz @ K.T
		if np.linalg.norm(xp - xe) < model.iter_limit:
			break
		xp = xe
	return xe, P_xx

# 基于近似最小一乘估计的容积卡尔曼滤波器
def ALA_RCKF(model, x, z, P_xx, t):
	n = 2 * model.nx
	S_xx = np.linalg.cholesky(model.nx * P_xx)
	xs = np.block([-S_xx, S_xx]).T + x
	xps = np.block([[model.F(e, t)] for e in xs])
	xp = np.mean(xps, axis = 0)
	P_xx = xps.T @ xps / n - np.outer(xp, xp) + model.G @ model.Q @ model.G.T

	S_xx = np.linalg.cholesky(model.nx * P_xx)
	xps = np.block([-S_xx, S_xx]).T + xp
	zps = np.block([[model.H(e, t)] for e in xps])
	zp = np.mean(zps, axis = 0)

	inn = z - zp
	phi = np.sqrt(inn @ model.Ri @ inn.T)
	km = 1.345
	factor = phi if phi > km else 1
	R = factor * model.R
	P_zz = zps.T @ zps / n - np.outer(zp, zp) + R

	P_xz = xps.T @ zps / n - np.outer(xp, zp)
	K = P_xz @ np.linalg.inv(P_zz)
	xe = xp + K @ inn
	P_xx = P_xx - K @ P_zz @ K.T

	return xe, P_xx

# 基于近似最小一乘估计的迭代容积卡尔曼滤波器
def ALA_RICKF(model, x, z, P_xx, t): 
	m = 2 * model.nx
	S_xx = np.linalg.cholesky(model.nx * P_xx)
	xs = np.block([-S_xx, S_xx]).T + x
	xps = np.block([[model.F(e, t)] for e in xs])
	xp = np.mean(xps, axis = 0) # 预测状态
	P_xx = xps.T @ xps / m - np.outer(xp, xp) + model.G @ model.Q @ model.G.T # 预测状态误差协方差

	xe = xp
	g = 1 # 卡尔曼增益衰减因子
	niter = 1
	if t > model.iter_start: # 初始迭代抑制策略
		niter = model.niter 
	for i in range(niter):
		S_xx = np.linalg.cholesky(model.nx * P_xx)
		xps = np.block([-S_xx, S_xx]).T + xp
		# xp = np.mean(xps, axis = 0)
		zps = np.block([[model.H(e, t)] for e in xps])
		zp = np.mean(zps, axis = 0)

		inn = z - zp
		phi = np.sqrt(inn @ model.Ri @ inn.T)
		km = 1.345
		factor = phi if phi > km else 1
		R = factor * model.R
		P_zz = zps.T @ zps / m - np.outer(zp, zp) + R

		P_xz = xps.T @ zps / m - np.outer(xp, zp)
		K = P_xz @ np.linalg.inv(P_zz)
		xe = xe + g * K @ (z - zp).T
		P_xx = P_xx - K @ P_zz @ K.T
		g = model.eta * g
		xp = xe
	return xe, P_xx

# 基于近似最小一乘估计的平方根容积卡尔曼滤波器
def ALA_RICKF_SR(model, x, z, P_xx, t):
	m = 2 * model.nx
	S_xx = np.linalg.cholesky(P_xx)
	xs = np.block([-S_xx, S_xx]).T * np.sqrt(model.nx) + x
	xps = np.block([[model.F(e, t)] for e in xs])
	xp = np.mean(xps, axis = 0)
	Xs = (xps - xp).T / np.sqrt(m)
	S_xx = Tria(np.block([Xs, model.Qs]))

	xe = xp
	niter = 1
	if t > model.iter_start: # 初始迭代抑制策略
		niter = model.niter 
	for i in range(niter):
		xps = np.block([-S_xx, S_xx]).T * np.sqrt(model.nx) + xe
		zps = np.block([[model.H(e, t)] for e in xps])
		zp = np.mean(zps, axis = 0)
		Xps = (xps - xe).T / np.sqrt(m)
		Zps = (zps - zp).T / np.sqrt(m)

		inn = z - zp
		phi = np.sqrt(inn @ model.Ri @ inn.T)
		km = 1.345
		factor = phi if phi > km else 1
		R = factor * model.R
		Sr = np.linalg.cholesky(R)

		S_zz = Tria(np.block([Zps, Sr]))
		P_xz = Xps @ Zps.T
		K = P_xz / S_zz.T / S_zz
		xe = xe + K @ inn.T
		S_xx = Tria(np.block([Xps - K @ Zps, K @ Sr]))
		dx = xp - xe
		if np.linalg.norm(dx) < model.iter_limit:
			break
		xp = xe
	return xe, S_xx @ S_xx.T

# 基于Lamine - Mili鲁棒估计框架的迭代容积卡尔曼滤波器
def L_RICKF(model, x, z, P_xx, t):
	m = 2 * model.nx
	S_xx = np.linalg.cholesky(model.nx * P_xx)
	xs = np.block([-S_xx, S_xx]).T + x
	xps = np.block([[model.F(e, t)] for e in xs])
	xp = np.mean(xps, axis = 0)
	P_xx = xps.T @ xps / m - np.outer(xp, xp)  + model.G @ model.Q @ model.G.T

	xe = xp
	niter = 1
	if t > model.iter_start:
		niter = int(np.sqrt(model.niter))
	for i in range(niter):
		S_xx = np.linalg.cholesky(model.nx * P_xx)
		xps = np.block([-S_xx, S_xx]).T + xe
		zps = np.block([[model.H(e, t)] for e in xps])
		zp = np.mean(zps, axis = 0)

		PR = sl.block_diag(P_xx, model.R) # 联合协方差
		PR_SR = np.linalg.cholesky(PR)
		PR_SRI = np.linalg.inv(PR_SR)

		y = PR_SRI @ np.hstack([xp, z]).T
		yp = PR_SRI @ np.hstack([xe, zp]).T
		tau = 1.5
		C = []
		r = []
		for j in range(niter):
			r = y - yp
			rs = np.abs(r)
			W = np.diag([tau / e if e > tau else 1 for e in rs]) # Huber权矩阵

			dH_dx = model.JH(xe, t) # 量测函数在x处的Jacobian
			C = PR_SRI @ np.vstack([np.eye(model.nx), dH_dx]) # 批回归函数对x的导数

			dx = np.linalg.inv(C.T @ W @ C) @ C.T @ W @ (y - yp)
			if np.linalg.norm(dx) < model.iter_limit:
				break
			xe = xe + dx
			yp = PR_SRI @ np.hstack([xe, model.H(xe, t)]).T

		P_xx = 1.0369 * np.linalg.inv(C.T @ C) # 当tau = 1.5时，有文献证明前面的期望之比为1.0369
		xp = xe
	return xe, P_xx

# 基于Lamine - Mili鲁棒估计框架的容积卡尔曼滤波器
def GM_CKF(model, x, z, P_xx, t):
	n = 2 * model.nx
	S_xx = np.linalg.cholesky(model.nx * P_xx)
	xs = np.block([-S_xx, S_xx]).T + x
	
	xps = np.block([[model.F(e, t)] for e in xs])
	xp = np.mean(xps, axis = 0)

	P_xx = xps.T @ xps / n - np.outer(xp, xp) + model.Q
	S_xx = np.linalg.cholesky(model.nx * P_xx)
	xps = np.block([-S_xx, S_xx]).T + xp
	zps = np.block([[model.H(e, t)] for e in xps])
	zp = np.mean(zps, axis = 0)

	PR = sl.block_diag(P_xx, model.R) # 联合协方差
	PRs = np.linalg.cholesky(PR).T
	PRsi = np.linalg.inv(PRs)

	xe = xp
	y = PRsi @ np.hstack([xp, z]).T
	yp = PRsi @ np.hstack([xe, zp]).T
	tau = 1.5
	C = []
	r = []
	for j in range(6):
		r = y - yp
		r = r / (1.4826 * np.median(np.abs(r)))
		rs = np.abs(r)
		W = np.diag([tau / e if e > tau else 1 for e in rs])

		dH_dx = model.JH(xe, t)
		C = PRsi @ np.vstack([np.eye(model.nx), dH_dx])

		dx = np.linalg.inv(C.T @ W @ C) @ C.T @ W @ (y - yp)
		if np.linalg.norm(dx) < 1e-2:
			break
		xe = xe + dx
		yp = PRsi @ np.hstack([xe, model.H(xe, t)]).T

	P_xx = np.linalg.inv(C.T @ C)
	return xe, P_xx

# 标准粒子滤波器
def PF(model, xs, z, P_xx, t):
	N = len(xs)
	ws = np.zeros(N)
	for i in range(N):
		xs[i] = model.FP(xs[i], t) # 按状态转移方程更新粒子，其中含有随机性，这是粒子滤波的活力所在
		ws[i] = model.P_z__x(z, xs[i], t) # 以似然概率作权重
	ws = ws / np.sum(ws)
	ids = rs.residual_resample(ws)
	xs = xs[ids]
	xe = np.mean(xs, axis = 0)
	P_xx = (xs - xe).T @ (xs - xe) / N # + 1e-9
	return xe, P_xx, xs

# 基于高斯滤波器预处理的粒子滤波器，配合前面的各种非线性高斯滤波器使用
class KPF():
	def __init__(self, XKF):
		self.XKF = XKF

	def filter(self, model, xs, z, P_xx, t):
		N = len(xs)
		ws = np.zeros(N)
		for i in range(N):
			x0 = xs[i]
			x1, P = self.XKF(model, xs[i], z, P_xx, t) # 对粒子对应状态做卡尔曼滤波
			xs[i] = np.random.multivariate_normal(x1, P) # 在以估计状态为中心的高斯分布上采样出新粒子
			prior = model.P_x1__x0(xs[i], x0, t)
			likely = model.P_z__x(z, xs[i], t)
			proposal = st.multivariate_normal.pdf(xs[i], x1, P)
			ws[i] = likely * prior / proposal
		ws = ws / np.sum(ws)
		ids = rs.residual_resample(ws)
		xs = xs[ids]
		xe = np.sum(xs, axis = 0) / N
		P_xx = (xs - xe).T @ (xs - xe) / N + 1e-9 * np.eye(model.nx)
		return xe, P_xx, xs
