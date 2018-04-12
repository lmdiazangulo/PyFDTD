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
L         = 10.0
dx        = 0.05
finalTime = L/c0*2
cfl       = .8

totalFieldBox_lim = np.array((L*1./4., L*3./4.))
totalFieldBox_len = totalFieldBox_lim[1] - totalFieldBox_lim[0]

delay  = 8e-9
spread = 2e-9

xini = 20
xfin = 40
yini = 30
yfin = 60


def gaussianFunction(x, x0, spread):
    # Cast function to a numpy array
    x = x*np.ones(1, dtype=float)
    gaussian = np.zeros(x.size, dtype=float)
    for i in range(x.size):
        gaussian[i] = math.exp( - math.pow(x[i] - x0, 2) /
                                  (2.0 * math.pow(spread, 2)) )
    return gaussian

# ==== Inputs / Pre-processing ================================================ 
# ---- Problem definition -----------------------------------------------------
L         = 10.0
dx        = 0.1
dy        = 0.1
finalTime = L/c0*5
cfl       = .99

gridEX = np.linspace(0,      L,        num=L/dx+1, endpoint=True)
gridEY = np.linspace(0,      L,        num=L/dy+1, endpoint=True)
gridHX = np.linspace(dx/2.0, L-dx/2.0, num=L/dx,   endpoint=True)
gridHY = np.linspace(dx/2.0, L-dx/2.0, num=L/dy,   endpoint=True)

# ---- Materials --------------------------------------------------------------

# ---- Boundary conditions ----------------------------------------------------
 
# ---- Sources ----------------------------------------------------------------
"""
# Initial field
spread   = 1.0
center   = (L/2.0, L/2.0)

initialH = np.zeros((gridHX.size, gridHY.size))
for i in range(gridHX.size):
    for j in range(gridHY.size):
        initialH[i,j] = math.exp( 
            - ((gridHX[i]-center[0])**2 + (gridHY[j]-center[1])**2) /
            math.sqrt(2.0) / spread)
"""
 
# ---- Output requests --------------------------------------------------------
samplingPeriod = 0.0
 
# ==== Processing =============================================================
# ---- Solver initialization --------------------------------------------------
dt = cfl * dx / c0 / math.sqrt(2)
numberOfTimeSteps = int( finalTime / dt )

if samplingPeriod == 0.0:
    samplingPeriod = dt 
nSamples  = int( math.floor(finalTime/samplingPeriod) )
probeH    = np.zeros((gridHX.size, gridHY.size, nSamples))
probeTime = np.zeros(nSamples) 

exOld = np.zeros((gridEX.size, gridEY.size), dtype=float)
exNew = np.zeros((gridEX.size, gridEY.size), dtype=float)
eyOld = np.zeros((gridEX.size, gridEY.size), dtype=float)
eyNew = np.zeros((gridEX.size, gridEY.size), dtype=float)
hzOld = np.zeros((gridHX.size, gridHY.size), dtype=float)
hzNew = np.zeros((gridHX.size, gridHY.size), dtype=float)

if 'initialH' in locals():
    hzOld = initialH

# Determines recursion coefficients
cEx = dt / eps0 / dx
cEy = dt / eps0 / dy
cHx = dt / mu0  / dx
cHy = dt / mu0  / dy

# ---- Time integration -------------------------------------------------------
print('--- Processing starts---')
tic = time.time();

t = 0.0
for n in range(numberOfTimeSteps):
    # --- Updates E field ---
    for i in range(1, gridEX.size-1):
        for j in range(1, gridEY.size-1):
            exNew[i][j] = exOld[i][j] + cEy * (hzOld[i][j] - hzOld[i  ][j-1])
            eyNew[i][j] = eyOld[i][j] - cEx * (hzOld[i][j] - hzOld[i-1][j  ])
    
    for j in range(yini, yfin+1):           
        eyNew[xini][j] += gaussianFunction(dt*n, delay, spread)
        eyNew[xfin][j] -= gaussianFunction(
                dt*n + (xfin-xini)*dx/c0, delay, spread)    


    # E field boundary conditions
    
    # PEC
    exNew[ :][ 0] = 0.0;
    exNew[ :][-1] = 0.0;
    eyNew[ 0][ :] = 0.0;
    eyNew[-1][ :] = 0.0;  

    # --- Updates H field ---
    for i in range(gridHX.size):
        for j in range(gridHX.size):
            hzNew[i][j] = hzOld[i][j] - cHx * (eyNew[i+1][j  ] - eyNew[i][j]) +\
                                        cHy * (exNew[i  ][j+1] - exNew[i][j])
        
    for j in range(yini, yfin+1):           
        hzNew[xini-1][j] += gaussianFunction(dt*n, delay, spread)
        hzNew[xfin-1][j] -= gaussianFunction(
                dt*n + (xfin-xini)*dx/c0, delay, spread)  
      
    # --- Updates output requests ---
    probeH[:,:,n] = hzNew[:,:]
    probeTime[n] = t
    
    # --- Updates fields and time 
    exOld[:] = exNew[:]
    eyOld[:] = eyNew[:]
    hzOld[:] = hzNew[:]
    t += dt
    print ("Time step: %d of %d" % (n, numberOfTimeSteps-1))

tictoc = time.time() - tic;
print('--- Processing finished ---')
print("CPU Time: %f [s]" % tictoc)

# ==== Post-processing ========================================================

# --- Creates animation ---
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
#ax = plt.axes(xlim=(gridE[0], gridE[-1]), ylim=(-1.1, 1.1))
ax.set_xlabel('X coordinate [m]')
ax.set_ylabel('Y coordinate [m]')
line = plt.imshow(probeH[:,:,0], animated=True, vmin=-0.5, vmax=0.5)
timeText = ax.text(0.02, 0.95, '', transform=ax.transAxes)

def init():
    line.set_array(probeH[:,:,0])
    timeText.set_text('')
    return line, timeText

def animate(i):
    line.set_array(probeH[:,:,i])
    timeText.set_text('Time = %2.1f [ns]' % (probeTime[i]*1e9))
    return line, timeText

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=nSamples, interval=50, blit=True)

plt.show()

print('=== Program finished ===')
