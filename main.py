import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.constants
import time
import matplotlib.animation as animation

# ==== Preamble ===============================================================
c0   = scipy.constants.speed_of_light
mu0  = scipy.constants.mu_0
eps0 = scipy.constants.epsilon_0
imp0 = math.sqrt(mu0 / eps0)

def gaussianFunction(t, t0, spread):
    return np.exp(- np.power(t-t0, 2) / (2.0 * np.power(spread, 2)) )

def averagedNorm(fAnal, fExp):
    vec = np.power(fAnal[:]-fExp[:],2)
    return np.sqrt(np.sum(vec)/np.size(vec))
 
# ==== Inputs / Pre-processing ================================================ 
# ---- Problem definition -----------------------------------------------------
L         = 250.0
dx        = 0.1
finalTime = L/c0*0.8
cfl       = 0.99
dt = cfl * dx / c0

gridE = np.linspace(0,      L,        num=L/dx+1, endpoint=True)
gridH = np.linspace(dx/2.0, L-dx/2.0, num=L/dx,   endpoint=True)

# Initial field
spread = 1/math.sqrt(2.0)
waveCenter = 5.0
initialE =   gaussianFunction(gridE, waveCenter, spread)
initialH = + gaussianFunction(gridH, waveCenter + c0*dt/2, spread) / imp0

# ---- Output requests --------------------------------------------------------
samplingPeriod = 0.0
 
# ==== Processing =============================================================
# ---- Solver initialization --------------------------------------------------

numberOfTimeSteps = int( finalTime / dt )

if samplingPeriod == 0.0:
    samplingPeriod = dt 
nSamples  = int( math.floor(finalTime/samplingPeriod) )
probeE            = np.zeros((gridE.size, nSamples))
probeH            = np.zeros((gridH.size, nSamples))
probeETheoretical = np.zeros((gridE.size, nSamples))
probeTime         = np.zeros(nSamples) 
errorNorm         = np.zeros(nSamples)

eOld = np.zeros(gridE.size)
eNew = np.zeros(gridE.size)
hOld = np.zeros(gridH.size)
hNew = np.zeros(gridH.size)
if 'initialE' in locals():
    eOld[:] = initialE[:]
if 'initialH' in locals():
    hOld[:] = initialH[:]

# Determines recursion coefficients
cE = dt / eps0 / dx
cH = dt / mu0  / dx

# ---- Time integration -------------------------------------------------------
print('--- Processing starts---')
tic = time.time()

w = 2*math.pi * 100e6
k = c0 / w

eOld[ 0] = 0.0
eOld[-1] = 0.0

t = 0.0
for n in range(numberOfTimeSteps):
    # --- Updates E field ---
    # for i in range(1, gridE.size-1):
    #    eNew[i] = eOld[i] + cE * (hOld[i-1] - hOld[i])
    eNew[1:-1] = eOld[1:-1] + cE * (hOld[:-1]-hOld[1:])

    # # PMC
    # eNew[ 0] = eOld[0] - 2*cE*hOld[0] 
    # eNew[-1] = eOld[-1] + 2*cE*hOld[-1] 
    
    #PEC
    eNew[ 0] = 0.0
    eNew[-1] = 0.0

    # --- Updates H field ---
    # for i in range(gridH.size):
    #    hNew[i] = hOld[i] + cH * (eNew[i] - eNew[i+1])
    hNew[:] = hOld[:] + cH * (eNew[:-1]-eNew[1:])
    
    # H field boundary conditions        
    # --- Updates output requests ---
    probeE[:,n]            = eNew[:]
    probeH[:,n]            = hNew[:]
    probeTime[n]           = t
    
    # --- Updates fields and time 
    eOld[:] = eNew[:]
    hOld[:] = hNew[:]
    t += dt
 
    probeETheoretical[:,n] = gaussianFunction(gridE, waveCenter + c0*t, spread)
    errorNorm[n] = averagedNorm(probeETheoretical[:,n], probeE[:,n])

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
line1,    = ax1.plot([], [], '--', markersize=1)
timeText1 = ax1.text(0.02, 0.95, '', transform=ax1.transAxes)

ax2 = fig.add_subplot(2, 2, 2)
ax2 = plt.axes(xlim=(gridE[0], gridE[-1]), ylim=(-1.1, 1.1))
ax2.grid(color='gray', linestyle='--', linewidth=.2)
line2,    = ax2.plot([], [], '-', markersize=1)
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
    line2.set_data(gridE, probeETheoretical[:,i])
    timeText2.set_text('Time = %2.1f [ns]' % (probeTime[i]*1e9))
    return line1, timeText1, line2, timeText2

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=nSamples, interval=50, blit=True)

plt.show()

print('=== Program finished ===')