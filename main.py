import numpy as np
import math
import scipy.constants
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ==== Preamble ===============================================================
c0   = scipy.constants.speed_of_light
eps0 = scipy.constants.epsilon_0
mu0  = scipy.constants.mu_0

def gaussianFunction(x, x0, spread):
    gaussian = np.zeros(x.size)
    for i in range(x.size):
        gaussian[i] = math.exp( - math.pow(x[i] - x0, 2) /
                                  (2 * math.pow(spread, 2)) )
    return gaussian

# ==== Inputs / Pre-processing ================================================ 
# ---- Problem definition -----------------------------------------------------
L         = 10.0
dx        = 0.1
finalTime = L/c0*2
cfl       = 1.0

grid = np.linspace(0, L, num=L/dx, endpoint=True)

# ---- Materials --------------------------------------------------------------

# ---- Boundary conditions ----------------------------------------------------
 
# ---- Sources ----------------------------------------------------------------
# Initial field
spread = 0.5
initialE = gaussianFunction(grid, L/2, spread)

# Plane wave illumination
totalFieldBox = (L*1/4, L*3/4)
delay  = 8e-9
spread = 2e-9
 
# ---- Output requests --------------------------------------------------------
samplingPeriod = 0.0
 
# ==== Processing =============================================================
# ---- Solver initialization --------------------------------------------------
dt = cfl * dx / c0
numberOfTimeSteps = int( finalTime / dt )

if samplingPeriod == 0.0:
    samplingPeriod = dt 
nSamples  = int( math.floor(finalTime/samplingPeriod) )
probeE    = np.zeros((grid.size, nSamples))
probeTime = np.zeros(nSamples) 

eOld = np.zeros(grid.size)
eNew = eOld
hOld = np.zeros(grid.size-1)
hNew = hOld
if 'initialE' in locals():
    eOld = initialE
if 'initialH' in locals():
    hOld = initialH

# totalFieldLeft = np.searchSorted(grid, totalFieldBox()[0])

# Determines recursion coefficients
cE = dt / eps0 / dx
cH = dt / mu0  / dx

# ---- Time integration -------------------------------------------------------
print('--- Processing starts---')
tic = time.time();

t = 0.0
for n in range(numberOfTimeSteps):
    # --- Updates E field ---
    for i in range(2, grid.size-1):
        eNew[i] = eOld[i] + cE * (hOld[i-1] - hOld[i])
     
    # E field boundary conditions
    # Sources
    # TODO

    # PEC
    eNew[ 0] = 0.0;
    eNew[-1] = 0.0;
    
    # PMC
    # TODO
    
    # Mur ABC
    # TODO

    # --- Updates H field ---
    for i in range(grid.size-1):
        hNew[i] = hOld[i] + cH * (eNew[i] - eNew[i+1])
    
    # H field boundary conditions
    # Sources
    # TODO
   
    # --- Updates output requests ---
    probeE[:,n]  = eNew
    probeTime[n] = t
    
    # --- Updates fields and time 
    eOld = eNew
    hOld = hNew
    t += dt

tictoc = time.time() - tic;
print('--- Processing finished ---')
print("CPU Time: %f [s]" % tictoc)

# ==== Post-processing ========================================================

# --- Creates animation ---
fig = plt.figure(figsize=(8,4))
ax = plt.axes(xlim=(grid[0], grid[-1]), ylim=(-1.1, 1.1))
ax.grid(color='gray', linestyle='--', linewidth=.2)
ax.set_xlabel('X coordinate [m]')
ax.set_ylabel('Electric field [V/m]')
line,    = ax.plot([], [], 'o', markersize=1)
timeText = ax.text(0.02, 0.95, '', transform=ax.transAxes)

def init():
    line.set_data([], [])
    timeText.set_text('')
    return line, timeText

def animate(i):
    line.set_data(grid, probeE[:,i])
    timeText.set_text('Time = %2.1f [ns]' % (probeTime[i]*1e9))
    return line, timeText

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=nSamples, interval=50, blit=True)

plt.show()

print('=== Program finished ===')
