"""
@author mdzik
 Class to handle python calls, see zalesak exaple and test
 !! This is part of test suite
"""

import numpy as np

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
    
    def getTime(self):
        return self.time


