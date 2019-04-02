import numpy as np
import math
import scipy.constants
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation

L = 10

def F_analitica(x,t):
    F_left = 0.5 * np.exp(-(x+t-L/2)**2)
    F_right = 0.5 * np.exp(-(x-t-L/2)**2)
    return F_left + F_right

Fig = plt.figure()

def Init():
    c_num = 0.01
    x = np.linspace(0,L,1000)
    y = F_analitica(x,0)
    plt.plot(x,y)

def Iter(i):
    Fig.clf()
    c_num = 0.01
    x = np.linspace(0,L,1000)
    y = F_analitica(x,c_num*i)
    plt.plot(x,y)

animation.FuncAnimation(Fig,Iter,frames=np.arange(1,1000),init_func=Init)
plt.show()