import numpy as np
import matplotlib.pyplot as plt

def plotCosine():
    plt.figure("Q1b_1190298")
    n = np.arange(0, 50, 1)
    x = np.cos(0.04*np.pi*n)+0.2*np.random.randn(len(n)) # x = cos(0.04*pi*n)+0.2*randn(1,length(n))
    plt.stem(n,x)
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    plt.axis([0,50,-1.5,1.5]) # plt.axis([xmin,xmax,ymin,ymax])
    plt.grid(True)
    plt.title("x(n) = cos(0.04*pi*n)+0.2*randn(1,length(n))")
    plt.show()
plotCosine()