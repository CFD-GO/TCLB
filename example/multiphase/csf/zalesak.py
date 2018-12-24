
import numpy as np
#import matplotlib.pyplot as plt

class CallPythonHelper:
    def __init__(self, *args):
        nofargs = args[0]
        self.offsets, self.time, self.lattice_size = args[1:nofargs]
        self.data = args[nofargs:]

    def getVector(self,idx):
        V = np.asarray(self.data[idx])
        shape = V.shape
        return np.asarray(self.data[idx]).reshape((shape[1],shape[0],3))

    def getScalar(self,idx):
        V = np.asarray(self.data[idx])
        shape = V.shape
        return np.asarray(self.data[idx]).reshape((shape[1],shape[0]))

    def getXY(self,scal_idx=0):
        V = np.asarray(self.data[scal_idx])
        shape = V.shape

        X = np.zeros([shape[1],shape[0]])
        Y = np.zeros([shape[1],shape[0]])
        for x in range(shape[0]):
            X[:,x] = self.offsets[0] + x

        for y in range(shape[1]):
            Y[y,:] =  self.offsets[1] + y
        return X,Y


def setSwirl(*args):
    cph = CallPythonHelper(*args)


    Fx = cph.getScalar(0)
    Fy = cph.getScalar(1)
    X,Y = cph.getXY(0)
    X = X - cph.lattice_size[0]/2.
    Y = Y - cph.lattice_size[1]/2.
    L = np.max(X)
    X = X/L
    Y = Y/L

    R = np.sqrt(X**2 + Y**2)
    phi = np.arctan2(X,Y)
    c = (L * np.pi / 5000) #*(1 - np.tanh(10*(R-0.8)))

#    Fx[:,:] = c*R*np.cos(phi)
#    Fy[:,:] = -c*R*np.sin(phi)

    Fx[:,:] = -c * Y
    Fy[:,:] = c * X

    #Fx[1:-1,1:-1]= 0
    #Fy[1:-1,1:-1]= 0
#    plt.contourf(X,Y,Fx**2 + Fy**2 )
 #   plt.colorbar()
  #  plt.streamplot(X,Y,Fx,Fy,density=0.6, color='k', linewidth=1)
#    plt.show()
    return 0
if __name__ == "__main__":
    Y = list()
    X = list()
    of = list()
    test(of, 0, X, Y)
