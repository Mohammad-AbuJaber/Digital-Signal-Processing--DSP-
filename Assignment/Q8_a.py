import numpy as np
import matplotlib.pyplot as plt

a = [1]
b = [1, -1]
n1 = np.arange(0, 23, 1)
def stepseq(n0,n1,n2):
    n = np.arange(n1,n2+1)
    x = np.zeros(len(n))
    nx = np.zeros(len(n))
    for i in range(len(n)):
        if n[i] >= n0:
            x[i] = 1
            nx[i] = n[i]
    return x,nx

x11,nx11 = stepseq(0,0,22) # x11 u[n-0]
x12,nx12 = stepseq(20,0,22) # x12 u[n-20]
x1 = 5*(x11 - x12) # x1 = 5*(u[n-0] - u[n-20])

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
y1 = filter(b,a,x1)

plt.figure("Q8a_1190298")
plt.stem(n1,y1)
plt.axis([-1,23,-6,6])
plt.xlabel('n')
plt.ylabel('y(n)')
plt.title('Output response for rectangular pulse ')
plt.show()
