import numpy as np
import matplotlib.pyplot as plt

def plotRealAndImaginaryPart():
    plt.figure('Q3_1190298')
    n = np.arange(-10, 11, 1)
    alpha = -0.1+0.3j
    x = np.exp(alpha*n)
    #####################################
    plt.subplot(2,2,1)
    plt.stem(n,np.real(x))
    plt.title('Real Part')
    plt.xlabel('n')
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    #####################################
    plt.subplot(2,2,2)
    plt.stem(n,np.abs(x))
    plt.title('Magnitude Part')
    plt.xlabel('n')
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    #####################################
    plt.subplot(2,2,3)
    plt.stem(n,np.imag(x))
    plt.title('Imaginary Part')
    plt.xlabel('n')
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    #####################################
    plt.subplot(2,2,4)
    plt.stem(n,(180/np.pi)*np.angle(x)) # convert to degree
    plt.title('Phase Part')
    plt.xlabel('n')
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    #####################################
    plt.show()

plotRealAndImaginaryPart()