import numpy as np
import matplotlib.pyplot as plt

n = np.arange(0,101)
x = np.cos(np.pi*n/2)
y = np.exp(1j*np.pi*n/4)*x

# plt.subplot(2,1,1)
# plt.stem(n, x)
# plt.xlabel('n')
# plt.ylabel('x[n]')
# plt.title('x[n] vs n')

# plt.subplot(2,1,2)
# plt.stem(n, y)
# plt.xlabel('n')
# plt.ylabel('y[n]')
# plt.title('y[n] vs n')

k = np.linspace(-2*np.pi,2*np.pi,401)
Y = np.zeros(401)

def plot_spectrum(Y):
    magY = np.abs(Y)
    angY = np.angle(Y)

    plt.figure("Q11_1190298")
    plt.subplot(2,1,1)
    plt.plot(k, magY)
    plt.xlabel('Frequency (rad/sample)')
    plt.ylabel('Magnitude')
    plt.title('Magnitude Spectrum')
    plt.grid()

    plt.subplot(2,1,2)
    plt.plot(k, angY)
    plt.xlabel('Frequency (rad/sample)')
    plt.ylabel('Angle (rad)')
    plt.title('Angle Spectrum')
    plt.grid()

    plt.subplots_adjust(hspace=0.5)
    plt.show()

for i in range(401):
    Y[i] = np.sum(y * np.exp(-1j*k[i]*n))
plot_spectrum(Y)
print(Y)
# Y = np.fft.fft(y, 401)
