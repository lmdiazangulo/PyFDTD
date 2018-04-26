import numpy as np
import math
import matplotlib.pyplot as plt

N = 10000
t = np.linspace(0.0, 10.0, N)
dt = t[1] - t[0]
f = np.zeros(t.size)

delay = 2.0;
spread = 0.005;

for i in range(t.size):
    f[i]  = math.exp(-(t[i] - delay)**2/spread**2/2.0)

plt.figure(1)
plt.plot(t, f)


g = np.fft.fft(f)
freq = np.fft.fftfreq(N, dt)


plt.figure(2)
plt.plot(freq, abs(g),'-')
# plt.axis([0, 100])

plt.show()