import numpy as np
import matplotlib.pyplot as plt
b = [1] # this is the coefficients of x[n]
a = [1, -1, 0.9] # these are the coefficients of y[n] to get y[n] = 0.9y[n-1] - y[n-2]
n= np.arange(-5, 120, 1)

x = np.zeros(len(n))
for i in range(len(n)):
    if n[i] == 0:
        x[i] = 1

def filter(b, a, x):
    y = np.zeros(len(x))
    for i in range(len(x)):
        for j in range(len(b)):
            if i-j >= 0:
                y[i] += b[j]*x[i-j]
        for j in range(1, len(a)):
            if i-j >= 0:
                y[i] -= a[j]*y[i-j]
        y[i] /= a[0]
    return y

y = filter(b, a, x)
plt.figure("Q8_1190298")
plt.subplot(211)
plt.stem(n, y)
plt.grid()
plt.xlabel('n')
plt.ylabel('h(n)')
plt.title('A) Impulse response')
 
x = np.zeros(len(n))
for i in range(len(n)):
    if n[i] >= 0:
        x[i] = 1

y = filter(b, a, x)
plt.subplot(212)
plt.stem(n, y)
plt.grid()
plt.xlabel('n')
plt.ylabel('s(n)')
plt.title('B) Step response')
plt.subplots_adjust(hspace=1)
plt.show()

############################################
# Zplane
############################################
def zplane(b,a):
    plt.figure("Q8_1190298")
    ax = plt.subplot(111)
    unit_circle = plt.Circle((0,0), radius=1, fill=False, color='black', ls='dashed')
    ax.add_patch(unit_circle)
    plt.axis('scaled')
    plt.axis([-1.5, 1.5, -1.5, 1.5])
    plt.grid(True)
    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    # get the zeros and poles
    zeros = np.roots(b)
    print("zeros: ", zeros)
    poles = np.roots(a)
    print("poles: ", poles)
    # plot the zeros
    plt.plot(np.real(zeros), np.imag(zeros), 'go', ms=10)
    # plot the poles
    plt.plot(np.real(poles), np.imag(poles), 'rx', ms=10)
    # set the plot title
    plt.title('C) Pole/Zero Plot')
    # print the zeros and poles on the plot
    for n in range(len(zeros)):
        plt.text(np.real(zeros[n]), np.imag(zeros[n]), "  z{0}".format(n), '= ' + str(round(zeros[n], 3)))
    for n in range(len(poles)):
        plt.text(np.real(poles[n]), np.imag(poles[n]), "  p{0}".format(n) + ' = ' + str(round(poles[n], 3)))
    plt.grid(True)
    plt.show()

zplane(b, a)