
import numpy as np
from CallPythonHelper import *
#import matplotlib.pyplot as plt

def setSwirl(*args):
    cph = CallPythonHelper(*args)


    Fx = cph.getScalar(0)
    Fy = cph.getScalar(1)
    X,Y = cph.getXY(0)
    X = X - np.max(X)/2
    Y = Y - np.max(Y)/2
    L = np.max(X)
    X = X/L
    Y = Y/L

    R = np.sqrt(X**2 + Y**2)
    phi = np.arctan2(X,Y)
    c = (L * np.pi / 5000) *(1 - np.tanh(10*(R-0.8)))

    Fx[:,:] = c*R*np.cos(phi)
    Fy[:,:] = -c*R*np.sin(phi)

    #Fx[:,:] = 0
    #Fy[:,:] = 0

    #Fx[1:-1,1:-1]= 0
    #Fy[1:-1,1:-1]= 0
#    plt.contourf(X,Y,Fx**2 + Fy**2 )
 #   plt.colorbar()
  #  plt.streamplot(X,Y,Fx,Fy,density=0.6, color='k', linewidth=1)
#    plt.show()
    return 0
if __name__ == "__main__":
    print "Use as TCLB <CallPython> input"
