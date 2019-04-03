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
    return np.exp(- np.power(t-t0, 2) / (2.0 * np.power(spread, 2)) )

#funcion que calcula el error 
#fanal=funcion analítica
#fexp=función calculada
def funcionerror(fanal,fexp):
    vec = np.power(fanal[:]-fexp[:],2)
    return np.sum(vec)/np.size(vec)

 
fig = plt.figure()
xdata = #vector x
ydata = #vector con los diferentes errores de mallado
plt.xscale("log") #pone escala logaritmica ejex, para cambiarla en el eje y: "plt.yscale("log")"
plt.plot(xdata,ydata)
plt.show()

# ==== Inputs / Pre-processing ================================================ 
# ---- Problem definition -----------------------------------------------------
L         = 10.0
dx        = 0.01
finalTime = L/c0*2
cfl       = .99

gridE = np.linspace(0,      L,        num=L/dx+1, endpoint=True)
gridH = np.linspace(dx/2.0, L-dx/2.0, num=L/dx,   endpoint=True)

# ---- Materials --------------------------------------------------------------

# ---- Boundary conditions ----------------------------------------------------
 
# ---- Sources ----------------------------------------------------------------
# Initial field
spread = 1/math.sqrt(2.0)
initialE = gaussianFunction(gridE, L/2, spread)
 
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

# Determines recursion coefficients
cE = dt / eps0 / dx
cH = dt / mu0  / dx

# ---- Time integration -------------------------------------------------------
print('--- Processing starts---')
tic = time.time()

w = 2*math.pi * 100e6
k = c0 / w

t = 0.0
for n in range(numberOfTimeSteps):
    # --- Updates E field ---
    #for i in range(1, gridE.size-1):
    #    eNew[i] = eOld[i] + cE * (hOld[i-1] - hOld[i])

    eNew[1:-1]=eOld[1:-1]+ cE * (hOld[:-1]-hOld[1:]) #Es lo mismo que el for de arriba pero gracias a la librerianympy
    #[1:-1] te dice empezar no desde el primero (0) sino desde el siguiente hasta el final
    #[0:-1] te dice empezar desde el principio hasta el final
    # PEC
    eNew[ 0] = 0.0
    eNew[-1] = 0.0
    
    # --- Updates H field ---
    #for i in range(gridH.size):
    #    hNew[i] = hOld[i] + cH * (eNew[i] - eNew[i+1])
    hNew[:]=hOld[:] +cH * (eNew[:-1]-eNew[1:])
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

#commit al repositorio local push al git hub y luego pull request al profe 