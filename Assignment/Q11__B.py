import numpy as np
import matplotlib.pyplot as plt

# Sample rate
fs = 401

# Create array of n values
n = np.arange(101)

# Calculate x[n] and y[n]
x = np.cos((np.pi * n)/2)
y = np.exp((1j * np.pi * n) / 4) * x


# Calculate DTFT at 401 frequencies between -2*pi and 2*pi
frequencies = np.linspace(-2*np.pi, 2*np.pi, fs)
dtft = np.zeros(fs, dtype=complex)
for k in range(fs):
    dtft[k] = sum(y * np.exp(-1j * frequencies[k] * n))

# Calculate magnitude and angle spectrum
magnitude_spectrum = np.abs(dtft)
angle_spectrum = np.angle(dtft)

# Plot magnitude and angle spectrum
plt.figure("Q11_Magnitude Spectrum_1190298")
plt.plot(frequencies, magnitude_spectrum)
plt.xlabel('Frequency (rad/s)')
plt.ylabel('Magnitude')
plt.title('Magnitude Spectrum')

plt.figure("Q11_Angle Spectrum_1190298")
plt.plot(frequencies, angle_spectrum)
plt.xlabel('Frequency (rad/s)')
plt.ylabel('Angle (rad)')
plt.title('Angle Spectrum')

plt.show()
