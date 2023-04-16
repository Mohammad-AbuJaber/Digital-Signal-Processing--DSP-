import numpy as np
import matplotlib.pyplot as plt

def sigshift(x,nx,k): # it will return the signal x[n-k]
    nx = nx + k
    return x,nx

def sigadd(y,ny,w , nw): # it will return the signal x[n] + w[n]
    #where w[n] is Gaussian sequence with mean 0 and variance 1
    #and x[n] is the signal x[n-k]
    y = y + w
    ny = nw
    return y,ny

def sigfold(x,nx): # it will return the signal x[-n]
    nx = -nx
    return x,nx

def conv(x,nx,h,nh): # it will return the convolution of x and h
    nyb = nx[0] + nh[0] # nyb is the starting point of y
    nye = nx[-1] + nh[-1] # nye is the ending point of y
    ny = np.arange(nyb, nye+1) # ny is the range of y
    y = np.convolve(x,h) # y is the convolution of x and h
    return y, ny


#[rxy,nrxy] = conv_m(y,ny,x,nx); % crosscorrelation
def crosscorrelation(x,nx,y,ny):
    y,ny = sigfold(y,ny) # obtain x[-n] 
    rxy,nrxy = conv(x,nx,y,ny) # calculate the convolution of x and y 
    return rxy,nrxy

def plotEverything(k,x,nx):
    plt.figure("Q6 with k = " + str(k))
    plt.subplot(3,1,1)
    plt.stem(range(-3,4),x)
    plt.title('x[n]')
    y,ny = sigshift(x,nx,k) # obtain x[n-k]
    w = np.random.randn(len(y))
    nw = ny # generate w[n]
    y,ny = sigadd(y,ny,w,nw) # obtain y[n] = x[n-k] + w[n]
    plt.subplot(3,1,2)
    plt.stem(range(-3 + k,4 + k),y) # the range is equal to the range of x plus k
    plt.title('y[n]')
    x,nx = sigfold(x,nx) # obtain x[-n]
    rxy,nrxy = crosscorrelation(x,nx,y,ny) # crosscorrelation 
    plt.subplot(3,1,3)
    plt.stem(range(-6 + k,7 + k),rxy) # the range is equal to the sum of the ranges of x and y
    for i in range(len(rxy)):
        plt.text(i-6 + k, rxy[i]+0.5, str(round((rxy[i]),2)))
    plt.title('rxy')
    plt.subplots_adjust(hspace=1)
    plt.show()

if __name__ == '__main__':
    x = [3, 11, 7, 0, -1, 4, 2]; 
    nx = np.arange(-3,4) # given signal x[n] nx = [-3 -2 -1  0  1  2  3]
    plotEverything(2,x,nx) # Q7.a: k = 2 ==> y[n] = x[n-2] + w[n]
    plotEverything(4,x,nx) # Q7.b: k = 3 ==> y[n] = x[n-3] + w[n]