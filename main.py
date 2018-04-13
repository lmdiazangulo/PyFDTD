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
    return math.exp(- math.pow(t-t0, 2) / (2.0 * math.pow(spread, 2)) )

# ==== Inputs / Pre-processing ================================================ 
# ---- Problem definition -----------------------------------------------------
L         = 10.0
dx        = 0.05
finalTime = L / c0 * 2.0
cfl       = .99

gridE = np.linspace(0,      L,        num=L/dx+1, endpoint=True)
gridH = np.linspace(dx/2.0, L-dx/2.0, num=L/dx,   endpoint=True)

# ---- Materials --------------------------------------------------------------
# PML
pmlStart = 3.0 / 4.0 * L
pmlSigmaE0 = 1e1
pmlSigmaH0 = pmlSigmaE0*mu0/eps0

# ---- Boundary conditions ----------------------------------------------------
 
# ---- Sources ----------------------------------------------------------------
# Initial field
spread = 1/math.sqrt(2.0)

initialE = np.zeros(gridE.size)
for i in range(initialE.size):
    initialE[i] = math.exp(- math.pow(gridE[i] - L/4.0, 2) / (2.0 * math.pow(spread, 2)) )

initialH = np.zeros(gridH.size)
for i in range(initialH.size):
    initialH[i] = math.exp(- math.pow(gridH[i] - L/4.0 - dx/2, 2) / (2.0 * math.pow(spread, 2)) ) / imp0

# Plane wave illumination
totalFieldBox = (L*1/8, L*7/8)
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
if 'initialE' in locals():
    hOld = initialH

totalFieldIndices = np.searchsorted(gridE, totalFieldBox)
shift = (gridE[totalFieldIndices[1]] - gridE[totalFieldIndices[0]]) / c0 

pmlIndex = np.searchsorted(gridE, 3.0*L/4.0)
dOld = np.zeros(gridE.size)
dNew = np.zeros(gridE.size)
bOld = np.zeros(gridH.size)
bNew = np.zeros(gridH.size)

# Determines recursion coefficients
cE = dt / eps0 / dx
cH = dt / mu0  / dx

# ---- Time integration -------------------------------------------------------
print('--- Processing starts---')
tic = time.time();

w = 2*math.pi * 100e6;
k = c0 / w;

t = 0.0
for n in range(numberOfTimeSteps):
    # --- Updates E field ---
    for i in range(1, gridE.size-1):
        eNew[i] = eOld[i] + cE * (hOld[i-1] - hOld[i])
     
    for i in range(pmlIndex, gridE.size-1):
        pmlSigmaE = pmlSigmaE0*pow((gridE[i]-pmlStart)/(L - pmlStart),3)
        eNew[i] = (2*eps0 - dt*pmlSigmaE)/(2*eps0 + dt*pmlSigmaE)*eOld[i] + \
                    2*dt/(2*eps0+dt*pmlSigmaE)/dx*(hOld[i-1] - hOld[i])

    # E field boundary conditions
    # Sources   
#     eNew[totalFieldIndices[0]] = eNew[totalFieldIndices[0]] + gaussianFunction(t, delay, spread)
#     eNew[totalFieldIndices[1]] = eNew[totalFieldIndices[1]] - gaussianFunction(t, delay+shift, spread)

    # PEC
    eNew[ 0] = 0.0;
    eNew[-1] = 0.0;
    
    # PMC
#     eNew[ 0] = eOld[ 0] - 2.0 * cE * hOld[ 0]
#     eNew[-1] = eOld[-1] + 2.0 * cE * hOld[-1]
    
    # Mur ABC
#     eNew[ 0] = eOld[ 1] + (c0*dt-dx)/(c0*dt+dx) * (eNew[ 1] - eOld[ 0])         
#     eNew[-1] = eOld[-2] + (c0*dt-dx)/(c0*dt+dx) * (eNew[-2] - eOld[-1]) 

    # --- Updates H field ---
    for i in range(gridH.size):
        hNew[i] = hOld[i] + cH * (eNew[i] - eNew[i+1])
        
    for i in range(pmlIndex-1, gridH.size):
        pmlSigmaH = pmlSigmaH0*pow((gridH[i]-pmlStart)/(L - pmlStart),3)
        hNew[i] = (2*mu0 - dt*pmlSigmaH)/(2*mu0 + dt*pmlSigmaH)*hOld[i] + \
                  2*dt/(2*mu0+dt*pmlSigmaH)*(eNew[i] - eNew[i+1])/dx
    
    # H field boundary conditions
    # Sources
#     hNew[totalFieldIndices[0]-1] = hNew[totalFieldIndices[0]-1] + gaussianFunction(t, delay, spread) / imp0
#     hNew[totalFieldIndices[1]-1] = hNew[totalFieldIndices[1]-1] - gaussianFunction(t, delay+shift, spread) / imp0
          
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