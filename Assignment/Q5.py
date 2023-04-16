import numpy as np
import matplotlib.pyplot as plt

def conv(x,nx,h,nh): # it will return the convolution of x and h
    nyb = nx[0] + nh[0] # nyb is the starting point of y
    nye = nx[-1] + nh[-1] # nye is the ending point of y
    ny = np.arange(nyb, nye+1) # ny is the range of y
    y = np.convolve(x,h) # y is the convolution of x and h
    return y, ny

def plotConv(x,nx,h,nh):
    plt.figure('Q5_1190298')
    y, ny = conv(x,nx,h,nh)
    plt.subplot(3,1,1)
    plt.stem(nx,x, 'r')
    plt.title('Input') 
    plt.xlabel('n'),plt.ylabel('Amplitude')
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')

    plt.subplot(3,1,2)
    plt.stem(nh,h, 'b')
    plt.title('impulse response') 
    plt.xlabel('n'),plt.ylabel('Amplitude')
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')

    plt.subplot(3,1,3)
    plt.stem(ny, y, 'k')
    plt.title('Output')
    plt.xlabel('n'),plt.ylabel('Amplitude')
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')

    plt.axis([-5, 45, -1, 10])
    plt.subplots_adjust(hspace=1)
    plt.show()

#x(n) = u(n) − u(n − 10)
x = np.array([1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0])
nx = np.arange(0,21)
h = 0.9**np.arange(0,21)
nh = np.arange(0,21)
plotConv(x,nx,h,nh)