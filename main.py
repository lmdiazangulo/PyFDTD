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
L         = 10
dx        = 0.05
finalTime = L/c0/4

grid = np.linspace(0, L, num=L/dx, endpoint=True);
cfl  = 1


# ---- Materials --------------------------------------------------------------

# ---- Boundary conditions ----------------------------------------------------
 
# ---- Sources ----------------------------------------------------------------
# Initial field
spread = 0.5;
initialE = analyticalGaussian(grid, 0.0, spread);

# Plane wave illumination
totalFieldBox = (math.floor(grid.size * 1/4), math.floor(grid.size * 3/4));
delay  = 8e-9;
spread = 2e-9;
 
# ---- Output requests --------------------------------------------------------
samplingPeriod = 0.0;
 
# ==== Processing =============================================================
# ---- Solver initialization --------------------------------------------------
# Initializes spatial semi-discretization.
dt = cfl*dx/c0

eOld = np.linspace(0, L, num=grid.size, endpoint=True)
eNew = eOld
hOld = np.linspace(0, L, num=grid.size, endpoint=False)
hNew = hOld
if 'initialE' in locals():
    eOld = initialE
if 'initialH' in locals():
    hOld = initialH

# Determines recursion coefficients
cE = dt / eps0 / dx
cH = dt / mu0  / dx

# ---- Time integration -------------------------------------------------------
print('--- Starts processing ---')
tic = time.time();

numberOfTimeSteps = finalTime / dt
for n in range(numberOfTimeSteps):
    t += dt
    # --- Updates E field ---
    for i in range(2, grid.size-1):
        eNew[i] = eOld[i] + cE * (hOld[i-1] - hOld[i])
     
    # E field boundary conditions
    # Sources
#     ez(excPoint,2) = ez(excPoint,2) + exp(- 0.5*((t-delay)/spread)^2);    
#     phaseShift = (x(scaPoint) - x(excPoint)) / c0;
#     ez(scaPoint,2) = ez(scaPoint,2) - exp(- 0.5*((t-delay-phaseShift)/spread)^2);
    
    # PEC
    ez(    1, 2) = 0;
    ez(cells, 2) = 0;
    
    # PMC
    ez(    1, 2)    = ez(    1,1) - 2*cE*hy(      1,1);
    ez(cells,2)=ez(cells,1) + 2*cE*hy(cells-1,1);
    
    # Mur ABC
    ez(1,2) = ez(2,1) + (c0*dt-dx)/(c0*dt+dx)*(ez(2,2) - ez(1,1));
    ez(cells,2) = ez(cells-1,1) + (c0*dt-dx)/(c0*dt+dx)*(ez(cells,2) - ez(cells-1,1)); 

    # --- Updates H field ---
    for i=1:cells-1:
        hy(i,2)=hy(i,1)+cH*(ez(i,2)-ez(i+1,2));
    
    # E field boundary conditions
    # Sources
    hy(excPoint,2) = hy(excPoint,2) +  exp(- 0.5*((t+dt/2-delay)/spread)^2)/eta0;
    hy(scaPoint,2) = hy(scaPoint,2) - exp(- 0.5*((t+dt/2-delay-phaseShift)/spread)^2)/eta0;

    # PEC
    ez(:,1)=ez(:,2);
    hy(:,1)=hy(:,2);
 
    # --- Updates output requests ---
    subplot(2,1,1);
    hold off;
    plot(x,ez(:,1));
    hold on;
    xAn = 0:0.001:L;
    plot(xAn,analyticalGaussian(xAn,t+dt/2,L,spread),'-r');
    axis([x(1) x(end) -1 1]);
    title(sprintf('FDTD Time = %.2f nsec',t*1e9))
    subplot(2,1,2);
    hold off;
    plot(x(1:end-1),hy(:,1));
    hold on;
    axis([x(1) x(end) -0.005 0.005]);
    pause(.0025);
    drawnow;

tictoc = time.time() - tic;
print('--- Final time reached ---')
print('CPU Time: %f', tictoc)

# ==== Post-processing ========================================================

# Eanalytical = analyticalGaussian(x,t+dt/2,L,spread);
# 
# fprintf('For dx=%e , L^2 error: %e\n', ...
#     dx, sum(abs(ez(:,2)-Eanalytical))/cells);