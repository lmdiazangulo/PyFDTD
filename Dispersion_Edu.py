import numpy as np
import math
import cmath
import scipy.constants
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numpy.linalg import norm

# ==== Preamble ===============================================================
c0   = scipy.constants.speed_of_light
mu0  = scipy.constants.mu_0
eps0 = scipy.constants.epsilon_0
imp0 = math.sqrt(mu0 / eps0)

def gaussianFunction(t, t0, spread):
    return math.exp(- math.pow(t-t0, 2) / (2.0 * math.pow(spread, 2)) )

# ==== Inputs / Pre-processing ================================================ 
# ---- Problem definition -----------------------------------------------------
L         = 20.0
dx        = 0.05
finalTime = L/c0*2
cfl       = .99


gridE = np.linspace(0,      L,        num=L/dx+1, endpoint=True)
gridH = np.linspace(dx/2.0, L-dx/2.0, num=L/dx,   endpoint=True)

# ---- Materials --------------------------------------------------------------

# ---- Boundary conditions ----------------------------------------------------
 
# ---- Sources ----------------------------------------------------------------
# Initial field
spread = 1/math.sqrt(2.0)
#initialE = gaussianFunction(gridE, L/2, spread)

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


#Material de Debye
#dep = 2.1
##taup = 0.7
#cp = (dep)/(2*taup)
#ap = -1/taup
cp=2e1
ap=4e1
kp = (1+ap*dt/2)/(1-ap*dt/2)
betap = (eps0*cp*dt)/(1-ap*dt/2)
epsr=2
eps=eps0*epsr
sigma = 2e-3
lim1 = 120
lim2 = 200 

l = (lim1 - lim2) * dx

eOld = np.zeros(gridE.size)
eNew = np.zeros(gridE.size)
hOld = np.zeros(gridH.size)
hNew = np.zeros(gridH.size)
jOld = np.zeros(gridE.size)
jNew = np.zeros(gridE.size)

er = np.zeros(numberOfTimeSteps)
ei = np.zeros(numberOfTimeSteps)
CR= np.zeros(numberOfTimeSteps)
CR_teorico= np.zeros(numberOfTimeSteps)
er_norm=np.zeros(numberOfTimeSteps)
ei_norm=np.zeros(numberOfTimeSteps)




if 'initialE' in locals():
    eOld = initialE
    
for i in range(1, gridE.size-1):
    jNew[i]=kp*jOld[i-1]+betap*((eOld[i]-eOld[i-1])/dt)   

totalFieldIndices = np.searchsorted(gridE, totalFieldBox)
shift = (gridE[totalFieldIndices[1]] - gridE[totalFieldIndices[0]]) / c0 

# Determines recursion coefficients
cE = dt / eps0 / dx
cH = dt / mu0  / dx

# ---- Time integration -------------------------------------------------------
print('--- Processing starts---')
tic = time.time();

w = 2*math.pi * 100e6;
#k = c0 / w;
freq = np.fft.fftfreq(numberOfTimeSteps, dt)




t = 0.0
for n in range(numberOfTimeSteps):
    # --- Updates E field ---
    for i in range(1, gridE.size-1):
        
        if i in range(lim1, lim2):
        
            A=(2*eps+2*np.real(betap)-sigma*dt)/(2*eps+2*np.real(betap)+sigma*dt)
            B=2*eps+2*np.real(betap)+sigma*dt
            C=(hOld[i-1] - hOld[i])-np.real((1+kp)*jOld[i])
            eNew[i] = A*eOld[i]+2*dt*C/B
            jNew[i]=kp*jOld[i]+betap*((eNew[i]-eOld[i])/dt)
     
        else:
            eNew[i] = eOld[i] + cE * (hOld[i-1] - hOld[i])
   
    # E field boundary conditions
    # Sources   
    eNew[totalFieldIndices[0]] = eNew[totalFieldIndices[0]] + gaussianFunction(t, delay, spread)
    if t <= 50e-9:
        er[n] = eNew[5]
        ei[n] = eNew[totalFieldIndices[0]]

    
    # Mur ABC
    eNew[ 0] = eOld[ 1] + (c0*dt-dx)/(c0*dt+dx) * (eNew[ 1] - eOld[ 0])         
    eNew[-1] = eOld[-2] + (c0*dt-dx)/(c0*dt+dx) * (eNew[-2] - eOld[-1]) 

    # --- Updates H field ---
    for i in range(gridH.size):
        hNew[i] = hOld[i] + cH * (eNew[i] - eNew[i+1])
    
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
    jOld[:] = jNew[:]
    t += dt
    
# Transformada de fourier:
er_fft = np.fft.fft(er)    
ei_fft = np.fft.fft(ei)


for i in range (numberOfTimeSteps):
    er_norm[i] = norm(er_fft[i])
    ei_norm[i] = norm(ei_fft[i])
    CR[i] = er_norm[i] / ei_norm[i]

    
#    print('Coeficiente de reflexion:', er_norm[i], ei_norm[i], CR[i] )
#    print('Coeficiente de reflexion:', ei_norm)
#print('Coeficiente de reflexion:', CR)
    

eta=np.sqrt(mu0/eps0)
for i in range(numberOfTimeSteps):
    aa = math.cos((freq[i]/c0) * l)
    bb = complex(0, eta * math.sin((freq[i]/c0) * l))
    cc = complex(0, (eta ** -1) * math.sin((freq[i]/c0) * l))
    dd = aa
    CR_teorico[i]=  (aa * eta + bb - cc * eta **2 - dd * eta) / (aa * eta + bb + cc * eta **2 + dd *eta)
print ('CR:', CR_teorico)

plt.figure(1)
plt.plot(freq, CR_teorico)
plt.plot(freq, CR)
plt.savefig("CR.png")
plt.show()



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
plt.axvline(lim1 * dx, linewidth=1, c='olive')
plt.axvline(lim2 * dx, linewidth=1, c='olive')
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

print('=== Program finished ===')