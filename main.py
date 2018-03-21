import numpy as np
import math
import scipy.constants
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ==== Preamble ===============================================================
# c0   = scipy.constants.speed_of_light
# eps0 = scipy.constants.epsilon_0
# mu0  = scipy.constants.mu_0
c0 = 3e8
mu0 = math.pi*4e-7;
eps0=1/(mu0*c0**2);

def gaussianFunction(x, x0, spread):
    gaussian = np.zeros(x.size)
    for i in range(x.size):
        gaussian[i] = math.exp( - math.pow(x[i] - x0, 2) /
                                  (2.0 * math.pow(spread, 2)) )
    return gaussian

# ==== Inputs / Pre-processing ================================================ 
# ---- Problem definition -----------------------------------------------------
L         = 10.0
dx        = 0.05
finalTime = L/c0*2
cfl       = 1.0

gridE = np.linspace(0,      L,        num=L/dx+1, endpoint=True)
gridH = np.linspace(dx/2.0, L-dx/2.0, num=L/dx,   endpoint=True)

# ---- Materials --------------------------------------------------------------

# ---- Boundary conditions ----------------------------------------------------
 
# ---- Sources ----------------------------------------------------------------
# Initial field
spread = 1/math.sqrt(2.0)
initialE = gaussianFunction(gridE, L/2, spread)

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
probeE    = np.zeros((gridE.size, nSamples))
probeH    = np.zeros((gridH.size, nSamples))
probeTime = np.zeros(nSamples) 

eOld = np.zeros(gridE.size)
eNew = np.zeros(gridE.size)
hOld = np.zeros(gridH.size)
hNew = np.zeros(gridH.size)
if 'initialE' in locals():
    eOld = initialE

# totalFieldLeft = np.searchSorted(gridE, totalFieldBox()[0])

# Determines recursion coefficients
cE = dt / eps0 / dx
cH = dt / mu0  / dx

# ---- Time integration -------------------------------------------------------
print('--- Processing starts---')
tic = time.time();

t = 0.0
for n in range(numberOfTimeSteps):
    # --- Updates E field ---
    for i in range(1, gridE.size-1):
        eNew[i] = eOld[i] + cE * (hOld[i-1] - hOld[i])
     
    # E field boundary conditions
    # Sources
#     ez(excPoint,2) = ez(excPoint,2) + exp(- 0.5*((t-delay)/spread)^2);
#     
#     phaseShift = (x(scaPoint) - x(excPoint)) / c0;
#     ez(scaPoint,2) = ez(scaPoint,2) - ...
#      exp(- 0.5*((t-delay-phaseShift)/spread)^2);

    # PEC
#     eNew[ 0] = 0.0;
#     eNew[-1] = 0.0;
    
    # PMC
#     eNew[ 0] = eOld[ 0] - 2.0 * cE * hOld[ 0];
#     eNew[-1] = eOld[-1] + 2.0 * cE * hOld[-1];
    
    # Mur ABC
    eNew[ 0] = eOld[ 1] + (c0*dt-dx)/(c0*dt+dx) * (eNew[ 1] - eOld[ 0])         
    eNew[-1] = eOld[-2] + (c0*dt-dx)/(c0*dt+dx) * (eNew[-1] - eOld[-2]) 

    # --- Updates H field ---
    for i in range(gridH.size):
        hNew[i] = hOld[i] + cH * (eNew[i] - eNew[i+1])
    
    # H field boundary conditions
    # Sources
#     hy(excPoint,2) = hy(excPoint,2) + ...
#          exp(- 0.5*((t+dt/2-delay)/spread)^2)/eta0;
#     hy(scaPoint,2) = hy(scaPoint,2) - ...
#          exp(- 0.5*((t+dt/2-delay-phaseShift)/spread)^2)/eta0;
      
    # --- Updates output requests ---
    probeE[:,n] = eNew[:]
    probeH[:,n] = hNew[:]
    probeTime[n] = t
    
    # --- Updates fields and time 
    eOld[:] = eNew[:]
    hOld[:] = hNew[:]
    t += dt

tictoc = time.time() - tic;
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
# ax2.set_xlabel('X coordinate [m]')
# ax2.set_ylabel('Magnetic field [T]')
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
