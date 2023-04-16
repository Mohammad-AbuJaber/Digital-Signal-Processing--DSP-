import numpy as np
import matplotlib.pyplot as plt

def findAndPlotConv(x,h):
    plt.figure('Q4_1190298')
    y = np.convolve(x,h)
    plt.subplot(3,1,1)
    plt.stem(range(-3,4),x) # x[n] = [3, 11, 7, 0, -1, 4, 2],  -3<=n<=3
    plt.ylabel('x(n)')
    plt.xlabel('n')
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    plt.subplot(3,1,2)
    plt.stem(range(-1,5),h) # h[n] = [2, 3, 0, -5, 2, 1],  -1<=n<=4
    plt.ylabel('h(n)')
    plt.xlabel('n')
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    plt.subplot(3,1,3)
    plt.stem(range(-4,8),y) # y[n] = [  6  31  47   6 -51  -5  41  18 -22  -3   8   2],  -4<=n<=7
    plt.ylabel('y(n)')
    plt.xlabel('n')
    for i in range(len(y)):
        plt.text(i-4, y[i]+0.5, str(y[i]))
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    print(y)
    plt.subplots_adjust(hspace=0.5)
    plt.show()

x = [3, 11, 7, 0, -1, 4, 2]
h = [2, 3, 0, -5, 2, 1]
findAndPlotConv(x,h)