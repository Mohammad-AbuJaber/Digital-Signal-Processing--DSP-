import matplotlib.pyplot as plt
def plotImpulse(lst,lowerBound=0,upperBound=0):
    plt.figure("Q1a_1190298")
    if lowerBound==0 and upperBound==0:
        lowerBound=min(lst)
        upperBound=max(lst)
    minvalue=min(lst)
    maxvalue=max(lst)        
    plt.stem(range(lowerBound,upperBound+1),lst)
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    plt.axis([lowerBound-0.5,upperBound+0.5,minvalue-0.5,maxvalue+0.5])
    plt.grid(True)
    plt.legend(['x(n)'])
    plt.title('2δ(n+2) - δ(n-4)')
    plt.show()

plotImpulse([0,0,0,2,0,0,0,0,0,-1,0],-5,5)