import numpy as np
import matplotlib.pyplot as plt
import pycwt as wavelet

# Step 1: Generate the time series for system 1
T1 = 5  # Period of the first sine wave
T2 = 30  # Period of the second sine wave

tvec = np.linspace(0, 100, 1000)
signal1 = np.sin((2 * np.pi / T1) * tvec) + np.random.randn(len(tvec)) * 0.1

# Step 2: Generate the time series for system 2
# Independent of system 1 up to time T
T = 50  # Time after which system 1 influences system 2
influence_factor = 0.5

signal2 = np.zeros_like(tvec)
for i, t in enumerate(tvec):
    if t < T:
        signal2[i] = np.sin((2 * np.pi / T2) * t) + np.random.randn() * 0.1
    else:
        signal2[i] = (1 - influence_factor) * (np.sin((2 * np.pi / T2) * t) + np.random.randn() * 0.1) \
                     + influence_factor * signal1[i]

# Step 3: Compute the wavelet transform of each series
mother_wavelet = wavelet.DOG(12)
dt = tvec[1] - tvec[0]

wave1, scales1, freqs1, _, _, _ = wavelet.cwt(signal1, dt,s0=2*dt, wavelet=mother_wavelet)
wave2, scales2, freqs2, _, _, _ = wavelet.cwt(signal2, dt,s0=2*dt, wavelet=mother_wavelet)

# Step 4: Compute the cross wavelet transform
cross_wavelet = wave1 * np.conj(wave2)

# Step 5: Visualize the results

# Power spectrum of signal1
plt.figure(figsize=(15, 10))

plt.subplot(3, 1, 1)
plt.contourf(tvec, 1/freqs1, np.abs(wave1)**2, 100)
plt.title('Wavelet Power Spectrum of Signal 1')
plt.ylabel('Period')
plt.yscale('log')
plt.colorbar(label='Power')
plt.gca().invert_yaxis()

# Power spectrum of signal2
plt.subplot(3, 1, 2)
plt.contourf(tvec, 1/freqs2, np.abs(wave2)**2, 100)
plt.title('Wavelet Power Spectrum of Signal 2')
plt.ylabel('Period')
plt.yscale('log')
plt.colorbar(label='Power')
plt.gca().invert_yaxis()

# Cross wavelet power spectrum
plt.subplot(3, 1, 3)
plt.contourf(tvec, 1/freqs1, np.abs(cross_wavelet), 100)
plt.title('Cross Wavelet Power Spectrum')
plt.xlabel('Time')
plt.ylabel('Period')
plt.yscale('log')
plt.colorbar(label='Power')
plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()
