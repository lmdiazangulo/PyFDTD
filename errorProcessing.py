import numpy as np
import matplotlib.pyplot as plt

resolution    = np.array([1.0,    0.5,    0.1,     0.05,    0.02])
computedError = np.array([1.5e-3, 1.6e-4, 2.09e-7, 1.30e-8, 3.31e-10])

plt.plot(resolution, np.sqrt(computedError), marker='o')
plt.xscale('log')
plt.yscale('log')
plt.grid(True, which='both')
plt.minorticks_on()
plt.xlabel("Space step (dx)")
plt.ylabel("Computed error")
plt.show()