import numpy as np
import matplotlib.pyplot as plt

# Generate an array of 501 evenly spaced values from 0 to pi
w = np.linspace(0, np.pi, 501)

# Compute X using the given formula
X = np.exp(1j*w) / (np.exp(1j*w) - 0.5*np.ones(501))

# Compute the magnitude, angle, real and imaginary parts of X
magX = np.abs(X)
angX = np.angle(X)
realX = np.real(X)
imagX = np.imag(X)

plt.figure("Q9_1190298")
# Plot the magnitude of X in the first subplot
plt.subplot(2, 2, 1)
plt.plot(w/np.pi, magX)
plt.grid()
plt.xlabel('frequency in pi units')
plt.title('Magnitude Part')
plt.ylabel('Magnitude')

# Plot the angle of X in the third subplot
plt.subplot(2, 2, 3)
plt.plot(w/np.pi, angX)
plt.grid()
plt.xlabel('frequency in pi units')
plt.title('Angle Part')
plt.ylabel('Radians')

# Plot the real part of X in the second subplot
plt.subplot(2, 2, 2)
plt.plot(w/np.pi, realX)
plt.grid()
plt.xlabel('frequency in pi units')
plt.title('Real Part')
plt.ylabel('Real')

# Plot the imaginary part of X in the fourth subplot
plt.subplot(2, 2, 4)
plt.plot(w/np.pi, imagX)
plt.grid()
plt.xlabel('frequency in pi units')
plt.title('Imaginary Part')
plt.ylabel('Imaginary')
plt.subplots_adjust(hspace=0.5)
plt.show()
