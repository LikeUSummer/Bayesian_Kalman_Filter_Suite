# 武汉理工大学 刘俊杰 2021
# https://www.zhihu.com/people/da-xia-tian-60
import numpy as np
import scipy.linalg as sl
import scipy.stats as st
import numdifftools as nd
import gamma

# 非稳态量测模型
class exp1():
    nx = 1
    nz = 1
    Q = np.array([[1]]) # 状态转移过程噪声协方差
    G = np.eye(nx) # Gamma矩阵
    R = np.array([[1e-2]]) # 量测噪声协方差矩阵
    r = R**0.5@np.ones(nz) # 量测噪声标准差数组
    x0 = np.array([1]) # 初始状态估计值
    P0 = Q # 初始状态估计误差协方差

    Qs = np.linalg.cholesky(Q)
    Ri = np.linalg.inv(R)
    Rs = np.linalg.cholesky(R)
    Rsi = np.linalg.inv(Rs)

    niter = 6 # 允许迭代滤波器的最大迭代次数
    iter_limit = 1e-3 # 迭代计算的容差
    eta = 1 # 迭代过程中卡尔曼增益的衰减因子
    iter_start = 0 # 开启迭代的时刻，用于初始迭代抑制策略

    alpha = 0.1 # 量测数据的污染率
    kappa = 50 # 量测数据中野值强度（量测噪声混合高斯分布的均方差之比）

    def __init__(self):
        self.F = lambda x,t: 0.5*x + np.sin(0.1*np.pi*t) + 1
        self.FP = lambda x,t: self.F(x,t) + np.random.multivariate_normal(np.zeros(self.nz),self.Q) # np.random.gamma(3,2) 
        self.H = lambda x,t: 5*x - 2 if t>30 else 0.2*x**2
        self.P_x1__x0 = lambda x1,x0,t: st.multivariate_normal.pdf(x1,x0,self.Q) # (x1-x0)**(6-1)*np.exp(-(x1-x0)/2)
        self.P_z__x = lambda z,x,t:st.multivariate_normal.pdf(z,self.H(x,t),self.R)

        self.JF = nd.Jacobian(self.F)
        self.JH = nd.Jacobian(self.H)

    def HP(self,x,t):
        ns = np.random.rand(self.nz)
        for i in range(self.nz):
            if ns[i] < self.alpha:
                ns[i] = np.random.normal(0,self.kappa*self.r[i])
            else:
                ns[i] = np.random.normal(0,self.r[i])
        return self.H(x,t) + ns

# 单变量非平稳增长模型
class exp2():
    nx = 1
    nz = 1
    Q = np.array([[1]])
    G = np.eye(nx)
    R = np.array([[1e-3]])
    r = R**0.5@np.ones(nz)
    x0 = np.array([0.1])
    P0 = Q

    Qs = np.linalg.cholesky(Q)
    Qi = np.linalg.inv(Q)
    Ri = np.linalg.inv(R)
    Rs = np.linalg.cholesky(R)
    Rsi = np.linalg.inv(Rs)

    niter = 9
    iter_limit = 1e-3
    eta = 1
    iter_start = 0

    alpha = 0.1 # 污染率
    kappa = 50 # 野值强度

    def __init__(self):
        self.F = lambda x,t:0.5*x + 25*x/(1+x**2) + 8*np.cos(1.2*t)
        self.FP = lambda x,t:self.F(x,t) + np.random.multivariate_normal(np.zeros(self.nx),self.Q)
        self.H = lambda x,t:x**3/200
        self.P_x1__x0 = lambda x1,x0,t: st.multivariate_normal.pdf(x1,x0,self.Q)
        self.P_z__x = lambda z,x,t:st.multivariate_normal.pdf(z,self.H(x,t),self.R)

        self.JF = nd.Jacobian(self.F)
        self.JH = nd.Jacobian(self.H)
    
    def HP(self,x,t):
        ns = np.random.rand(self.nz)
        for i in range(self.nz):
            if ns[i] < self.alpha:
                ns[i] = np.random.normal(0,self.kappa*self.r[i])
            else:
                ns[i] = np.random.normal(0,self.r[i])
        return self.H(x,t) + ns

# 具有短时模型突变的单变量非平稳增长模型
class exp3():
    nx = 1
    nz = 1
    Q = np.array([[1]])
    G = np.eye(1)
    R = np.array([[1e-3]]) 
    r = R**0.5@np.ones(nz)
    x0 = np.array([0.1])
    P0 = Q

    Qs = np.linalg.cholesky(Q)
    Qi = np.linalg.inv(Q)
    Ri = np.linalg.inv(R)
    Rs = np.linalg.cholesky(R)
    Rsi = np.linalg.inv(Rs)

    niter = 25
    iter_limit = 1e-3
    eta = 1
    
    alpha = 0.1
    kappa = 50

    def __init__(self):
        self.FP = lambda x,t:self.F(x,t) + np.random.multivariate_normal(np.zeros(self.nx),self.Q)
        self.H = lambda x,t:x**3/200
        self.P_x1__x0 = lambda x1,x0,t: st.multivariate_normal.pdf(x1,x0,self.Q)
        self.P_z__x = lambda z,x,t:st.multivariate_normal.pdf(z,self.H(x,t),self.R)

        self.JF = nd.Jacobian(self.F)
        self.JH = nd.Jacobian(self.H)

    def F(self,x,t):
        if t>50 and t<56:
            return 2*x + x/(1+x)
        else:
            return -0.5*x + 25*x/(1+x**2) + 8*np.cos(1.2*t)
    
    def HP(self,x,t):
        ns = np.random.rand(self.nz)
        for i in range(self.nz):
            if ns[i] < self.alpha:
                ns[i] = np.random.normal(0,self.kappa*self.r[i])
            else:
                ns[i] = np.random.normal(0,self.r[i])
        return self.H(x,t) + ns

# 无人船目标跟踪模型
class exp4():
    nx = 6
    nz = 3
    dt = 1
    I3 = np.eye(3)
    Z3 = np.zeros((3,3))
    FM = np.block([[I3,dt*I3],[Z3,I3]])
    G = np.block([[dt**2/2*I3],[dt*I3]])
    Q = np.eye(3)*0.1**2
    R = np.diag([50,0.5 * np.pi / 180.0,0.5 * np.pi / 180.0])**2
    r = R**0.5@np.ones(nz)
    x0 = np.array([8000, 11000, 2000, -50, -100, 0])
    P0 = np.eye(nx)*1e6

    Qs = np.linalg.cholesky(Q)
    Ri = np.linalg.inv(R)
    Rs = np.linalg.cholesky(R)
    Rsi = np.linalg.inv(Rs)

    niter = 3
    iter_limit = 1e-3
    eta = 1

    alpha = 0.15
    kappa = 200

    def F(self,x,t):
        return self.FM@x

    def FP(self,x,t):
        return self.F(x,t) + self.G@np.random.multivariate_normal(np.zeros(3),self.Q)

    def H(self,x,t):
        return np.array([np.sqrt(x[0]**2+x[1]**2+x[2]**2),np.arctan2(x[1],x[0]),np.arctan2(x[2],np.sqrt(x[0]**2+x[1]**2))])

    def HP(self,x,t):
        ns = np.random.rand(self.nz)
        for i in range(self.nz):
            if ns[i] < self.alpha:
                ns[i] = np.random.normal(0,self.kappa*self.r[i])
            else:
                ns[i] = np.random.normal(0,self.r[i])
        return self.H(x,t) + ns

    JH = nd.Jacobian(lambda x,t: np.array([np.sqrt(x[0]**2+x[1]**2+x[2]**2),np.arctan2(x[1],x[0]),np.arctan2(x[2],np.sqrt(x[0]**2+x[1]**2))]) )

# 高速船目标跟踪
class exp5():
    nx = 15
    nz = 6

    dt = 0.01
    I3 = np.eye(3)
    Z3 = np.zeros((3,3))
    k = np.array([1,1,1])
    omega = np.array([0.8,0.8,0.8])
    zeta = np.array([0.3,0.3,0.3])
    M = np.array([[748.7,0,0],[0,189.1,93.8],[0,39.6,660.4]])
    Mi = np.linalg.inv(M)
    D = np.array([[12.3,0,0],[0,59.7,3],[0,3,7.1]])
    MiD = Mi@D
    Aw = np.block([[Z3,I3],[-np.diag(omega**2),-np.diag(2*zeta*omega)]])
    B = np.block([[Z3],[Z3],[Z3],[Mi],[Z3]])
    Ew = np.block([[Z3],[np.diag(k)]])
    Eb = np.eye(3)

    Q = 1e-2*np.diag([1e-4,1e-4,1e-4,1e-5,1e-5,1e-5])
    G = np.block([[Ew,np.zeros((6,3))],[Z3,Z3],[Z3,Z3],[Z3,Eb]])
    C = np.block([Z3,I3,I3,Z3,Z3])
    R = 1e-2*np.diag([1e-4,1e-4,1e-4,1e-5,1e-5,1e-5])
    r = R**0.5@np.ones(nz)
    x0 =np.zeros(nx) # np.random.randn(nx)*0.5
    P0 = np.diag([1e-3]*nx)

    Qs = np.linalg.cholesky(Q)
    Ri = np.linalg.inv(R)
    Rs = np.linalg.cholesky(R)
    Rsi = np.linalg.inv(Rs)

    niter = 4
    iter_limit = 1e-5
    eta = 1

    alpha = 0.1
    kappa = 100

    def F(self,x,t):
        c = np.cos(x[5])
        s = np.sin(x[5])
        J = np.array([[c,-s,0],[s,c,0],[0,0,1]])
        MiJ = self.M@J.T
        Z3 = np.zeros((3,3))
        f = np.block([[self.Aw,np.zeros((6,9))],[Z3,Z3,Z3,J,Z3],[Z3,Z3,Z3,self.MiD,MiJ],[Z3,Z3,Z3,Z3,0.001*self.I3]])
        return x + (f@x+self.B@np.array([5,np.sin(t),0.001]))*self.dt
    
    def FP(self,x,t):
        return self.F(x,t) + self.G@np.random.multivariate_normal(np.zeros(6),self.Q)
    
    def H(self,x,t):
        return x[3:9]
    
    def HP(self,x,t):
        ns = np.random.rand(self.nz)
        for i in range(self.nz):
            if ns[i] < self.alpha:
                ns[i] = np.random.normal(0,self.kappa*self.r[i])
            else:
                ns[i] = np.random.normal(0,self.r[i])
        return self.H(x,t) + ns
    JH = nd.Jacobian(lambda x,t: x[3:9])

# 二维机动目标跟踪
class exp6():
    nx = 4
    nz = 2

    dt = 1
    FM = np.array([[1,dt,0,0],[0,1,0,0],[0,0,1,dt],[0,0,0,1]])
    G = np.array([[0.5*dt**2,0],[dt,0],[0,0.5*dt**2],[0,dt]])
    q = np.array([[dt**4/4,dt**3/2],[dt**3/2,dt**2]])*0.1
    Q = np.diag([9,4])
    R = np.diag([25,1e-4])
    r = R**0.5@np.ones(nz)
    x0 = np.array([12000, -250, 8000, 100])
    P0 = np.eye(nx)*1e7

    # Qs = np.linalg.cholesky(Q)
    Ri = np.linalg.inv(R)
    Rs = np.linalg.cholesky(R)
    Rsi = np.linalg.inv(Rs)

    niter = 4
    iter_limit = 1e-3
    eta = 1

    alpha = 0.15
    kappa = 100

    def F(self,x,t):
        return self.FM@x
    def FP(self,x,t):
        return self.F(x,t) + self.G@np.random.multivariate_normal(np.zeros(2),self.Q)
    
    def H(self,x,t):
        r2 = x[0]**2+x[2]**2
        return np.array([np.sqrt(r2),np.arctan2(x[2],x[0])])

    def HP(self,x,t):
        ns = np.random.rand(self.nz)
        for i in range(self.nz):
            if ns[i] < self.alpha:
                ns[i] = np.random.normal(0,self.kappa*self.r[i])
            else:
                ns[i] = np.random.normal(0,self.r[i])
        return self.H(x,t) + ns

# 再入弹道目标跟踪
class exp7():
    nx = 7
    nz = 3

    dt = 0.1
    g = 3.986005e14
    phi = np.array([[1,dt],[0,1]])
    Phi = sl.block_diag(phi,phi,phi,1)
    tau = np.array([[dt**2/2,dt]]).T
    K = np.block([[sl.block_diag(tau,tau,tau)],[0,0,0]])
    G = np.eye(nx)
    theta = np.array([[dt**3/3,dt**2/2],[dt**2/2,dt]])
    Q = sl.block_diag(5*theta,5*theta,5*theta,5*dt)
    R = np.diag([100,0.017,0.017])**2
    r = R**0.5@np.ones(nz)
    x0 = np.array([232000,-1837,232000,-1837,90000,-1500,4000])
    P0 = np.diag([100,50,100,50,100,50,200])**2
    
    Qs = np.linalg.cholesky(G@Q@G.T)
    Ri = np.linalg.inv(R)
    Rs = np.linalg.cholesky(R)
    Rsi = np.linalg.inv(Rs)

    niter = 3
    iter_limit = 1e-3
    eta = 1

    alpha = 0.1
    kappa = 100

    def rho(self,h):
        if h < 9144:
            return 1.227*np.exp(-1.093e-4*h)
        else:
            return 1.745*np.exp(-1.49e-4*h)

    def F(self,x,t):
        zr = x[4]+6371000
        r = np.sqrt(x[0]**2+x[2]**2+zr**2)
        v = np.sqrt(x[1]**2+x[3]**2+x[5]**2)
        h = r - 6371000
        w = -1/(2*x[6])*v*self.rho(h)
        u = self.g/r**3
        return self.Phi@x + self.K@np.array([w*x[1]-u*x[0],w*x[3]-u*x[2],w*x[5]-u*zr])

    def FP(self,x,t):
        return self.F(x,t) + np.random.multivariate_normal(np.zeros(self.nx),self.Q)

    def H(self,x,t):
        return np.array([np.sqrt(x[0]**2+x[2]**2+x[4]**2),np.arctan2(x[4],np.sqrt(x[0]**2 + x[2]**2)),np.arctan2(x[2], x[0])])

    def HP(self,x,t):
        ns = np.random.rand(self.nz)
        for i in range(self.nz):
            if ns[i] < self.alpha:
                ns[i] = np.random.normal(0,self.kappa*self.r[i])
            else:
                ns[i] = np.random.normal(0,self.r[i])
        return self.H(x,t) + ns
    
    JH = nd.Jacobian(lambda x,t: np.array([np.sqrt(x[0]**2+x[2]**2+x[4]**2),np.arctan2(x[4],np.sqrt(x[0]**2+x[2]**2)),np.arctan2(x[2],x[0])]))
