# 武汉理工大学 刘俊杰 2021
# https://www.zhihu.com/people/da-xia-tian-60
import numpy as np
def gamma(alpha, beta):
    if alpha==1:
        x = -np.log(1-np.random.random())/beta
        return
    flag=0
    if alpha<1:
        flag=1
        alpha=alpha+1
    gamma=alpha-1
    eta=np.sqrt(2.0*alpha-1.0)

    c=.5-np.arctan(gamma/eta)/np.pi
    aux=-.5
    y = 0
    while aux<0:
        y=-.5
        while y<=0:
            u=np.random.random();  
            y = gamma + eta * np.tan(np.pi*(u-c)+c-.5)
        v=-np.log(np.random.random())
        aux=v+np.log(1.0+((y-gamma)/eta)**2)+gamma*np.log(y/gamma)-y+gamma

    if flag==1:
        x = y/beta*(np.random.random())**(1.0/(alpha-1))
    else:
        x = y/beta
    return x
