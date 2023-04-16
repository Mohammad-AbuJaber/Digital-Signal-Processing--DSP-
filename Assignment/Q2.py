import numpy as np
import matplotlib.pyplot as plt

# g(t) = cos(2*pi*f1*t) + 0.125 * cos(2*pi*2*f2*t)
# plot g[n]  for one second with sampling frequency fs

def plotCosine():
    # sampling frequencies
    fs1 = 50
    fs2 = 30
    fs3 = 20
    # Figure name
    plt.figure('Q2_1190298')
    # frequencies
    f1 = 5
    f2 = 15
    # cosine function with different sampling frequencies
    g1 = np.cos(2*np.pi*f1*np.arange(0, 1, 1/fs1))
    g2 = np.cos(2*np.pi*f2*np.arange(0, 1, 1/fs2))
    g3 = np.cos(2*np.pi*f2*np.arange(0, 1, 1/fs3))
    # time axis for each sampling frequency
    t1 = np.arange(0, 1, 1/fs1)
    t2 = np.arange(0, 1, 1/fs2)
    t3 = np.arange(0, 1, 1/fs3)
    # plot g[n] for each sampling frequency
    plt.subplot(3,1,1)
    plt.stem(t1,g1)
    plt.title('g[n] with fs = ' + str(fs1) + ' Hz')
    plt.xlabel('n')
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    plt.subplot(3,1,2)
    plt.stem(t2,g2)
    plt.title('g[n] with fs = ' + str(fs2) + ' Hz')
    plt.xlabel('n')
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    plt.subplot(3,1,3)
    plt.stem(t3,g3)
    plt.title('g[n] with fs = ' + str(fs3) + ' Hz')
    plt.xlabel('n')
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    # edit spacing between subplots
    plt.subplots_adjust(hspace=0.5)
    plt.show()

plotCosine()