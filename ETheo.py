import numpy as np
import math
import scipy.constants
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation

c0   = scipy.constants.speed_of_light
mu0  = scipy.constants.mu_0
eps0 = scipy.constants.epsilon_0
imp0 = math.sqrt(mu0 / eps0)

L = 10
dx        = 0.01
finalTime = L/c0*2
cfl       = .99

dt = cfl * dx / c0
numberOfTimeSteps = int( finalTime / dt )
gridSize = int(L / dx)

probeTime = np.zeros(numberOfTimeSteps) 
probeE = np.zeros((gridSize,numberOfTimeSteps))
gridE = np.zeros(gridSize)

def F_analitica(x,t):
    F_left = 0.5 * np.exp(-(x+1e9*t-L/2)**2)
    F_right = 0.5 * np.exp(-(x-1e9*t-L/2)**2)
    return F_left + F_right

t = 0.0

for n in range(numberOfTimeSteps):
    probeTime[n] = t
    x = 0.0
    for i in range(gridSize):
        probeE [i,n] = F_analitica (x,t)
        gridE [i] = x
        x += dx
    t += dt


# --- Creates animation ---
fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(1, 2, 1)
ax1 = plt.axes(xlim=(gridE[0], gridE[-1]), ylim=(-1.1, 1.1))
ax1.grid(color='gray', linestyle='--', linewidth=.2)
ax1.set_xlabel('X coordinate [m]')
ax1.set_ylabel('Field')
line1,    = ax1.plot([], [], 'o', markersize=1)
timeText1 = ax1.text(0.02, 0.95, '', transform=ax1.transAxes)

# ax2 = fig.add_subplot(2, 2, 2)
# ax2 = plt.axes(xlim=(gridE[0], gridE[-1]), ylim=(-1.1, 1.1))
# ax2.grid(color='gray', linestyle='--', linewidth=.2)
# ax2.set_xlabel('X coordinate [m]')
# ax2.set_ylabel('Magnetic field [T]')
# line2,    = ax2.plot([], [], 'o', markersize=1)
# timeText2 = ax2.text(0.02, 0.95, '', transform=ax2.transAxes)

def init():
    line1.set_data([], [])
    timeText1.set_text('')
#    line2.set_data([], [])
#    timeText2.set_text('')
    return line1, timeText1#, line2, timeText2

def animate(i):
    line1.set_data(gridE, probeE[:,i])
    timeText1.set_text('Time = %2.1f [ns]' % (probeTime[i]*1e9))
#    line2.set_data(gridH, probeH[:,i]*100)
#    timeText2.set_text('Time = %2.1f [ns]' % (probeTime[i]*1e9))
    return line1, timeText1#, line2, timeText2

anim = animation.FuncAnimation(fig, animate, init_func=init)

plt.show()