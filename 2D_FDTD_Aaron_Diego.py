import numpy as np
import matplotlib.pyplot as plt
#import scipy.constants
import math
from mpl_toolkits import mplot3d
import matplotlib.animation as animation

#Parámetros iniciales
Ly =10
Lx=10
T=25
K=500
M=100
N=100
dx = Lx/M
dy = Ly/N
dt = T/K

#Hacemos el espacio descreto
lin_x = np.linspace(0,      Lx,        M+1, endpoint=True)
lin_y = np.linspace(0,      Ly,        N+1, endpoint=True)
X, Y = np.meshgrid(lin_x, lin_y)

#Condiciones iniciales
spread = 0.5
H0 = np.exp(-(((X-5)**2)+(Y-5)**2)/spread)
Ex0 = 0.0*X
Ey0 = 0.0*Y

#Hacemos la figura inicial
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, H0, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_title('surface');
ax.set_xlabel('Eje x')
ax.set_ylabel('Eje y')
ax.set_zlabel('H')

#Declaramos variables de los campos
ExOld = Ex0
ExNew = np.zeros((len(X), len(Y)))
EyOld = Ey0
EyNew = np.zeros((len(X), len(Y)))
probeH    = np.zeros(((K, len(X), len(Y))))
hOld = H0
hNew = np.zeros((len(X), len(Y)))

#Calculamos con cada iteración los datos nuevos
for n in range(K):
    #Actualizar los datos
    ExNew[1:-1,1:-1] = ExOld[1:-1,1:-1] + (dt/dy)*(hOld[1:-1,2:] - hOld[1:-1,:-2])
    EyNew[1:-1,1:-1] = EyOld[1:-1,1:-1] - (dt/dx)*(hOld[2:,1:-1] - hOld[:-2,1:-1])
    hNew[1:-1,1:-1]= hOld[1:-1,1:-1] + (ExNew[1:-1,2:] - ExNew[1:-1,:-2]) - (EyNew[2:,1:-1] - EyNew[:-2,1:-1])

    #Condiciones de Frontera (reflectante)
    ExNew[:,0] = ExOld[:,0] + (dt/dy)*(hOld[:,1] - hOld[:,N-1])
    EyNew[0,:] = EyOld[0,:] - (dt/dx)*(hOld[1,:] - hOld[M-1,:])
    ExNew[:,N] = ExOld[:,N] + (dt/dy)*(hOld[:,1] - hOld[:,N-1])
    EyNew[M,:] = EyOld[M,:] - (dt/dx)*(hOld[1,:] - hOld[M-1,:])
    hNew[0,:]= hOld[0,:] + (ExNew[0,:] - ExNew[0,:]) - (EyNew[1,:] - EyNew[M-1,:])
    hNew[:,0]= hOld[:,0] + (ExNew[:,1] - ExNew[:,N-1]) - (EyNew[:,0] - EyNew[:,0])
    hNew[M,:]= hOld[M,:] + (ExNew[M,:] - ExNew[M,:]) - (EyNew[1,:] - EyNew[M-1,:])
    hNew[:,N]= hOld[:,N] + (ExNew[:,N] - ExNew[:,N-1]) - (EyNew[:,N] - EyNew[:,N])

    #Actualizar los datos para la proxima iteración
    ExOld[:,:] = ExNew[:,:]
    EyOld[:,:] = EyNew[:,:]
    hOld[:,:] = hNew[:,:]
    probeH[n,:,:] = hNew[:,:]

    #Hacer el plot
def animate(i):

    ax.collections.clear()
    ax.plot_surface(X, Y, probeH[i], rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
    return

anim = animation.FuncAnimation(fig, animate,
                               frames=K, interval=50, blit=False)

plt.show()
