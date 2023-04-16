import numpy as np
import matplotlib.pyplot as plt

def plotXtilde():
    plt.figure("Q1c_1190298")
    n = np.arange(-10, 10, 1) # n = [-10,-9,...,9]
    x = [5,4,3,2,1]
    xtilde = np.tile(x,4) # xtilde = [5,4,3,2,1,5,4,3,2,1,5,4,3,2,1,5,4,3,2,1]
    plt.stem(n,xtilde)
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    plt.axis([-10,9,0,6]) # plt.axis([xmin,xmax,ymin,ymax])
    plt.grid(True)
    plt.title('xtilde(n)')
    plt.show()
plotXtilde()