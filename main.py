import numpy as np
import math
import scipy.constants
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numpy.core.fromnumeric import size

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
finalTime = L/c0*2
cfl       = .99
TAM=2;
gridE = np.linspace(0,      L,        num=TAM*L/dx+1, endpoint=True)
gridH = np.linspace(dx/2.0, L-dx/2.0, num=TAM*L/dx,   endpoint=True)


# ---- Materials --------------------------------------------------------------

# ---- Boundary conditions ----------------------------------------------------
 
# ---- Sources ----------------------------------------------------------------
# Initial field
#spread = 1/math.sqrt(2.0)
# initialE = gaussianFunction(gridE, L/2, spread)

# Plane wave illumination
#totalFieldBox = (L*1/8, L*7/8)
totalFieldBox = (L*1/100,L*7/8)
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

totalFieldIndices = np.searchsorted(gridE, totalFieldBox)
#shift = (gridE[totalFieldIndices[1]] - gridE[totalFieldIndices[0]]) / c0 

shift=0

# Determines recursion coefficients
cE = dt / eps0 / dx
cH = dt / mu0  / dx

# ---- Time integration -------------------------------------------------------
print('--- Processing starts---')
tic = time.time()

w = 2*math.pi * 100e6
k = c0 / w
beta = w*np.sqrt(mu0*eps0)

t = 0.0
E_ExactaV= np.zeros((gridE.size, nSamples))
E_Exacta = np.zeros(gridE.size)
H_exactaV= np.zeros((gridH.size, nSamples))

for n in range(numberOfTimeSteps):
    # --- Updates E field ---

    for i in range(1, gridE.size-1):
        eNew[i] = eOld[i] + cE * (hOld[i-1] - hOld[i])
        E_Exacta[i] = gaussianFunction(gridE[i], c0*t, spread*c0)
        
    E_ExactaV[:,n] = E_Exacta[:]

    # E field boundary conditions
    # Sources   
    eNew[totalFieldIndices[0]] = eNew[totalFieldIndices[0]] + gaussianFunction(t, delay, spread)
#    eNew[0] = eNew[0] + gaussianFunction(t, delay, spread)
#     eNew[totalFieldIndices[1]] = eNew[totalFieldIndices[1]] - gaussianFunction(t, delay+shift, spread)

    # PEC
#    eNew[ 0] = 0.0;
#    eNew[-1] = 0.0;
    
    # PMC
#    eNew[ 0] = eOld[ 0] - 2.0 * cE * hOld[ 0]
#    eNew[-1] = eOld[-1] + 2.0 * cE * hOld[-1]
    
    # Mur ABC
    eNew[ 0] = eOld[ 1] + (c0*dt-dx)/(c0*dt+dx) * (eNew[ 1] - eOld[ 0])         
    eNew[-1] = eOld[-2] + (c0*dt-dx)/(c0*dt+dx) * (eNew[-2] - eOld[-1]) 

    # Periodic
#    eNew[ 0] = eOld[ 0] + cE * (hOld[ -1] - hOld[ 0])         
#    eNew[ -1] = eOld[ -1] + cE * (hOld[ -2] - hOld[ -1])

    # --- Updates H field ---
    H_exacta=[ ]
    for i in range(gridH.size):
        hNew[i] = hOld[i] + cH * (eNew[i] - eNew[i+1])
        H_exacta.append(gaussianFunction(gridH[i], c0*t, 1/math.sqrt(2.0)))
    
    H_exactaV[:,n] = H_exacta[:]
    # H field boundary conditions
    # Sources
    hNew[totalFieldIndices[0]-1] = hNew[totalFieldIndices[0]-1] + gaussianFunction(t, delay, spread) / imp0
#     hNew[totalFieldIndices[1]-1] = hNew[totalFieldIndices[1]-1] - gaussianFunction(t, delay+shift, spread) / imp0
          
    # --- Updates output requests ---
    probeE[:,n] = eNew[:]
    probeH[:,n] = hNew[:]
    probeTime[n] = t
    
    # --- Updates fields and time 
    eOld[:] = eNew[:]
    hOld[:] = hNew[:]
    t += dt

# --- guardar en ficheros
#np.savetxt("PMC_E%d.txt" %(TAM), ErrorE)
#np.savetxt("PMC_H%d.txt" %(TAM), ErrorH)

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

#plt.show()

plt.figure()

#plt.figure()
plt.plot(probeE[:,250])
plt.plot(E_ExactaV[:,100])


plt.show()

print('=== Program finished ===')