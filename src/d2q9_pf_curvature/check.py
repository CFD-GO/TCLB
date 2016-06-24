# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 14:13:06 2016

@author: mdzik
"""
from bearded_octo_wookie.CLB import *
import matplotlib.pyplot as plt
import numpy as np

import bearded_octo_wookie.lbm as lbm

from sympy.plotting import *
from sympy import *
import scipy.optimize as so

init_printing()

n=Symbol('n')
W=Symbol('w')
n0=Symbol('n0')

phi = -tanh(2*(n-n0)*W) / 2.


lap = diff(phi,n,2) + diff(phi,n) / n

grad = diff(phi,n)

grad = diff(phi,n)

pprint( simplify(expand(( lap - diff(phi,n,2))/grad)) )




fvti = '/home/mdzik/projekty/TCLB/output/phase-field-korteweg_VTK_P00_00001500.pvti'




    
vti = VTIFile.VTIFile(fvti, True)


PhaseField = vti.get('PhaseField', vector=False)
Curvature = vti.get('Curvature', vector=False)


X,Y = vti.getMeshGrid()

X = X - vti.dim[0] / 2
Y = Y - vti.dim[1] / 2


half = vti.dim[1] / 2

R = np.sqrt( X**2 + Y**2 )

array2mat = [{'ImmutableMatrix': np.matrix}, 'numpy']
laplace = lambdify([n,n0, W], lap, modules=array2mat)
phase = lambdify([n,n0, W], phi, modules=array2mat)
gradient = lambdify([n,n0, W], grad, modules=array2mat)

### find n00 and ww


(n00, ww), err = so.leastsq(lambda (C): phase(R,C[0], C[1])[half,:] - PhaseField[half,:], (256/2., .25) )
#ww = 0.025
#PhaseField = phase(R, n00, ww)

print n00, ww


laplace2 = np.zeros_like(R)

grad2 = np.zeros_like(R)
grad2_X = np.zeros_like(R)
grad2_Y = np.zeros_like(R)

for i in range(9):
    laplace2 = laplace2 + lbm.wp[i] * np.roll(np.roll(PhaseField,shift=lbm.e[i,0],axis=0),shift=lbm.e[i,1],axis=1) * 3

    grad2_X = grad2_X + lbm.W[i] * lbm.e[i,0] * np.roll(np.roll(PhaseField,shift=lbm.e[i,0],axis=0),shift=lbm.e[i,1],axis=1) * 3.
    grad2_Y = grad2_Y + lbm.W[i] * lbm.e[i,1] * np.roll(np.roll(PhaseField,shift=lbm.e[i,0],axis=0),shift=lbm.e[i,1],axis=1) * 3.


grad2 = np.sqrt(grad2_X**2 + grad2_Y**2)[half, :]

p2 = PhaseField[half, :]**2
grad_lengt = (4 * p2 - 1. ) * ww
curvature =   ( laplace2[half, :] - 2 * PhaseField[half, :] * (16 * p2 - 4. ) * ww**2 ) / grad_lengt



#plt.plot(laplace(R,n00, ww)[half, :] , '-')
#plt.plot(laplace(R,n00, ww)[half, :] - laplace2[half, :] , 'o')
#plt.plot(laplace2[half, :], 'x')
#plt.plot(Curvature[half, :], 'o')
#plt.plot(grad2[half, :], 'o-')
#plt.plot(grad_lengt )
dn = 10
plt.plot( curvature[n00-dn:n00+dn] )
plt.plot( np.ones_like(curvature)[n00-dn:n00+dn] *  R[half, n00-dn:n00+dn] )
##plt.plot(grad_lengt, 'o-')

#plt.figure()
#plt.imshow(PhaseField)
#plt.plot(phase(R,n00, ww)[half, :])
#plt.plot(PhaseField[half,:], 'o')


plt.show()
