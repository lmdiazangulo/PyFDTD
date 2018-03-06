import numpy as np
import math
import scipy.constants
import time

# ==== Preamble ===============================================================
c0   = scipy.constants.speed_of_light
eps0 = scipy.constants.epsilon_0
mu0  = scipy.constants.mu_0

def analyticalGaussian(coordinates, time, spread):
    return

# ==== Inputs / Pre-processing ================================================ 
# ---- Problem definition -----------------------------------------------------
L         = 10.0
dx        = 0.05
finalTime = L/c0/4
cfl       = 1.0

grid = np.linspace(0, L, num=L/dx, endpoint=True)

# ---- Materials --------------------------------------------------------------

# ---- Boundary conditions ----------------------------------------------------
 
# ---- Sources ----------------------------------------------------------------
# Initial field
spread = 0.5
initialE = analyticalGaussian(grid, 0.0, spread)

# Plane wave illumination
totalFieldBox = ( math.floor(grid.size * 1/4), math.floor(grid.size * 3/4) )
delay  = 8e-9
spread = 2e-9
 
# ---- Output requests --------------------------------------------------------
samplingPeriod = 0.0
 
# ==== Processing =============================================================
# ---- Solver initialization --------------------------------------------------
dt = cfl*dx/c0
numberOfTimeSteps = finalTime / dt

if samplingPeriod == 0.0:
    samplingPeriod = dt 
nSamples = finalTime/samplingPeriod
probeE    = np.zeros((grid.size, nSamples))
probeTime = np.zeros(nSamples) 

eOld = np.linspace(0, L, num=grid.size, endpoint=True)
eNew = eOld
hOld = np.linspace(0, L, num=grid.size, endpoint=False)
hNew = hOld
if 'initialE' in locals():
    eOld = initialE
if 'initialH' in locals():
    hOld = initialH

totalFieldBoxIndex = 
    ( np.searchSorted(grid, totalFieldBox()[0]),
      np.searchSorted(grid, totalFieldBox()[1]) );

# Determines recursion coefficients
cE = dt / eps0 / dx
cH = dt / mu0  / dx

# ---- Time integration -------------------------------------------------------
print('--- Starts processing ---')
tic = time.time();

t = 0.0
for n in range(numberOfTimeSteps):
    t += dt
    # --- Updates E field ---
    for i in range(2, grid.size-1):
        eNew[i] = eOld[i] + cE * (hOld[i-1] - hOld[i])
     
    # E field boundary conditions
    # Sources
#     if totalFieldBox in locals():
#         phaseShift = delay + grid[totalFieldBoxIndex()[0]] / c0s;
#         eNew[ totalFieldBoxIndex()[0] ] = 
#             eNew[ totalFieldBoxIndex()[0] ] + exp(- 0.5*((t-phaseShift)/spread)^2);    
#         ez(scaPoint,2) = 
#             ez(scaPoint,2) - exp(- 0.5*((t-delay-phaseShift)/spread)^2);
    
    # PEC
    eNew[ 0] = 0.0;
    eNew[-1] = 0.0;
    
    # PMC
#     ez(    1, 2)    = ez(    1,1) - 2*cE*hy(      1,1);
#     ez(cells,2)=ez(cells,1) + 2*cE*hy(cells-1,1);
    
    # Mur ABC
#     ez(1,2) = ez(2,1) + (c0*dt-dx)/(c0*dt+dx)*(ez(2,2) - ez(1,1));
#     ez(cells,2) = ez(cells-1,1) + (c0*dt-dx)/(c0*dt+dx)*(ez(cells,2) - ez(cells-1,1)); 

    # --- Updates H field ---
    for i in range(grid.size):
        hNew[i] = hOld[i] + cH * (eNew[i] - eNew[i+1])
    
    # E field boundary conditions
    # Sources
#     hy(excPoint,2) = hy(excPoint,2) +  exp(- 0.5*((t+dt/2-delay)/spread)^2)/eta0;
#     hy(scaPoint,2) = hy(scaPoint,2) - exp(- 0.5*((t+dt/2-delay-phaseShift)/spread)^2)/eta0;
   
    # Switches 
    eOld = eNew
    hOld = hNew
   
    # --- Updates output requests ---
    probeE

tictoc = time.time() - tic;
print('--- Final time reached ---')
print('CPU Time: %f', tictoc)

# ==== Post-processing ========================================================

# Eanalytical = analyticalGaussian(x,t+dt/2,L,spread);
# 
# fprintf('For dx=%e , L^2 error: %e\n', ...
#     dx, sum(abs(ez(:,2)-Eanalytical))/cells);