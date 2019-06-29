import numpy as np
import math
import scipy.constants
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ==== Preamble ===============================================================
c0   = scipy.constants.speed_of_light
mu0  = scipy.constants.mu_0
eps0 = scipy.constants.epsilon_0
imp0 = math.sqrt(mu0 / eps0)

def gaussianFunction(t, t0, spread):
    return np.sqrt(np.exp(- np.power(t-t0, 2) / (2.0 * np.power(spread, 2))))
 
# ==== Inputs / Pre-processing ================================================ 
# ---- Problem definition -----------------------------------------------------
L         = 10.0
L1        = 3.0
L2        = 4.0
dx        = 0.1
dx2       = 0.05
finalTime = L/c0*4.0
cfl       = 1.0 

# Total field parameters.
tFId  = (10, 130) # Total field indices
peak   = L/c0*0.5
spread = L/c0*0.1

gridE1 = np.linspace(0,      L1,        num=L1/dx,    endpoint=False)
gridE2 = np.linspace(L1,      L2+L1,    num=L2/dx2,   endpoint=False)
gridE3 = np.linspace(L1+L2,      L1*2+L2,   num=L1/dx+1,  endpoint=True)
gridE  = np.hstack([gridE1, gridE2, gridE3])

gridH  = (gridE[:-1]+gridE[1:])/2

# Initial field
# spread = 1/math.sqrt(2.0)
# initialE =   gaussianFunction(gridE, L/2, spread)
# initialH = + gaussianFunction(gridH, L/2+dx/2, spread) / imp0
 
# ---- Output requests --------------------------------------------------------
samplingPeriod = 0.0
 
# ==== Processing =============================================================
# ---- Solver initialization --------------------------------------------------
dt = cfl * dx2 / c0
numberOfTimeSteps = int( finalTime / dt )

if samplingPeriod == 0.0:
    samplingPeriod = dt 
nSamples  = int( math.floor(finalTime/samplingPeriod) )
probeE    = np.zeros((gridE.size, nSamples))
probeH    = np.zeros((gridH.size, nSamples))
probeTime = np.zeros(nSamples) 

eOld = np.zeros(gridE.size)
eNew = np.zeros(gridE.size)
hOld = np.zeros(gridH.size)
hNew = np.zeros(gridH.size)
if 'initialE' in locals():
    eOld = initialE
    hOld = initialH

# Determines recursion coefficients
cE = dt / eps0 / (gridH[1:]-gridH[:-1])
cH = dt / mu0  / (gridE[1:]-gridE[:-1])

# ---- Time integration -------------------------------------------------------
print('--- Processing starts---')
tic = time.time()

t = 0.0
for n in range(numberOfTimeSteps):
    # --- Updates E field ---
    # for i in range(1, gridE.size-1):
    #    eNew[i] = eOld[i] + cE * (hOld[i-1] - hOld[i])
    eNew[1:-1]=eOld[1:-1]+ cE * (hOld[:-1]-hOld[1:])

    # PMC
    # eNew[ 0] = eOld[0] - 2*cE*hOld[0] 
    # eNew[-1] = eOld[-1] + 2*cE*hOld[-1] 

    #PEC
    # eNew[0]=0.0
    eNew[-1]=0.0
    
    # Mur ABC
    eNew[0] = eOld[1] + (c0*dt - dx)/(c0*dt + dx)*(eNew[1]-eOld[0])

    # Total Field (electric)
    if 'tFId' in locals():
        delay = (gridE[tFId[1]] - gridE[tFId[0]]) / c0
        eNew[tFId[0]] += gaussianFunction(t        , peak, spread)*(dx/c0/dt)
        eNew[tFId[1]] -= gaussianFunction(t - delay, peak, spread)*(dx/c0/dt)

    # --- Updates H field ---
    hNew[:] = hOld[:] + cH * (eNew[:-1]-eNew[1:])
    
    # Total field (magnetic)
    if 'tFId' in locals():
        delay = (gridE[tFId[1]] - gridE[tFId[0]]) / c0
        hNew[tFId[0]-1] += gaussianFunction(t        , peak, spread)*(dx/c0/dt) / imp0
        hNew[tFId[1]-1] -= gaussianFunction(t - delay, peak, spread)*(dx/c0/dt) / imp0

    # H field boundary conditions        
    # --- Updates output requests ---
    probeE[:,n] = eNew[:]
    probeH[:,n] = hNew[:]
    probeTime[n] = t
    
    # --- Updates fields and time 
    eOld[:] = eNew[:]
    hOld[:] = hNew[:]
    t += dt

tictoc = time.time() - tic
print('--- Processing finished ---')
print("CPU Time: %f [s]" % tictoc)

# ==== Post-processing ========================================================
# --- Creates animation ---
fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(1, 2, 1)
ax1 = plt.axes(xlim=(gridE[0], gridE[-1]), ylim=(-1.1, 1.1))
ax1.grid(color='gray', linestyle='--', linewidth=.2)
ax1.set_xlabel('X coordinate [m]')
ax1.set_ylabel('Field')
line1,    = ax1.plot([], [], 'o', markersize=1)
timeText1 = ax1.text(0.02, 0.95, '', transform=ax1.transAxes)

ax2 = fig.add_subplot(2, 2, 2)
ax2 = plt.axes(xlim=(gridE[0], gridE[-1]), ylim=(-1.1, 1.1))
ax2.grid(color='gray', linestyle='--', linewidth=.2)
line2,    = ax2.plot([], [], 'o', markersize=1)
timeText2 = ax2.text(0.02, 0.95, '', transform=ax2.transAxes)

def init():
    line1.set_data([], [])
    timeText1.set_text('')
    line2.set_data([], [])
    timeText2.set_text('')
    return line1, timeText1, line2, timeText2

def animate(i):
    line1.set_data(gridE, probeE[:,i])
    timeText1.set_text('Time = %2.1f [ns]' % (probeTime[i]*1e9))
    line2.set_data(gridH, probeH[:,i]*100)
    timeText2.set_text('Time = %2.1f [ns]' % (probeTime[i]*1e9))
    return line1, timeText1, line2, timeText2

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=nSamples, interval=50, blit=True)

plt.show()

print('=== Program finished ===')

