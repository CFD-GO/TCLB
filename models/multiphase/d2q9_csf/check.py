# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 14:13:06 2016

@author: mdzik
"""
from CLB import *
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





half = 64
Wnum = 0.125


array2mat = [{'ImmutableMatrix': np.matrix}, 'numpy']
laplace = lambdify([n,n0, W], lap, modules=array2mat)
phase = lambdify([n,n0, W], phi, modules=array2mat)
gradient = lambdify([n,n0, W], grad, modules=array2mat)


#
#
#X,Y = np.meshgrid(np.arange(256)-128,np.arange(256)-128)
#R = np.sqrt(X*X+Y*Y)
#R0 = 64
#PHI = phase(R,R0,0.25)
#z = -PHI*2.
#plt.plot( np.arctanh(z)[128,:] / 2. / 0.25, 'o' )
#plt.plot(R[128,:] - R0)
#plt.show()
#sfsdf




for c, fvti in [
    #('k','/home/mdzik/projekty/TCLB/output/test1_2_VTK_P00_00001000.pvti'),
    #('r','/home/mdzik/projekty/TCLB/output/test1_omega1_VTK_P00_00006000.pvti'), 
    #('r','/home/mdzik/projekty/TCLB/output/test1_VTK_P00_00002000.pvti')    ,
    ('r','/home/mdzik/projekty/TCLB/output/test1_by_pf_VTK_P00_00006000.pvti')    
]:
    vti = VTIFile.VTIFile(fvti, True)
    PhaseField = vti.get('PhaseField', vector=False)
    #Curvature = vti.get('Curvature', vector=False)
    X,Y = vti.getMeshGrid()
    
    
    
    ### find n00 and ww
    (n00_l, ww_l, n00_r, ww_r), err = so.leastsq(lambda (C): -phase(X[half,:],C[0], C[1]) + phase(X[half,:],C[2],C[3]) -0.5  - PhaseField[half,:], (32., .25, 96., .25) )
    
    
    X = X - (n00_l + n00_r) / 2.
    Y = Y - half
    
    R = np.sqrt(X**2  + Y**2)
    
    (n00, ww), err = so.leastsq(lambda (C):  phase(R,C[0], C[1])[half,:] - PhaseField[half,:], (n00_l, ww_l) )
    print n00, ww
    #ww = 0.025
    #ww = 0.01
    #plt.imshow( phase(R, n00, ww) - PhaseField )
    
    #plt.colorbar()
    #plt.show()

    #plt.plot(phase(R, n00, ww)[half,:])
    #plt.plot(PhaseField[half,:])
    #plt.show()
    
    #plt.plot( phase(R,n00, ww)[half,:] , 'o')
    #plt.plot(PhaseField[half,:])
    #plt.plot(R[half,:] - n00)
    r_r0 = np.arctanh(-PhaseField * 2) /2 / Wnum
    
    r_r0 = np.where(np.isnan(r_r0), 0, r_r0)
    r_r0 = np.where(np.isinf(r_r0), 0, r_r0)
    r_r0 = np.where(np.isneginf(r_r0), 0, r_r0)
    #plt.plot(R[half,:] - n00)
    #plt.plot(r_r0[half,:])
    #plt.show()
    
    
    print (n00_l, ww_l, n00_r, ww_r)

    
    dn = 10
    
    laplace2 = np.zeros_like(R)
    
    grad2 = np.zeros_like(R)
    grad2_X = np.zeros_like(R)
    grad2_Y = np.zeros_like(R)
    
    for i in range(9):
        grad2_X = grad2_X + lbm.W[i] * lbm.e[i,0] * np.roll(np.roll(PhaseField,shift=-lbm.e[i,0],axis=0),shift=-lbm.e[i,1],axis=1) * 3.
        grad2_Y = grad2_Y + lbm.W[i] * lbm.e[i,1] * np.roll(np.roll(PhaseField,shift=-lbm.e[i,0],axis=0),shift=-lbm.e[i,1],axis=1) * 3.
    
    grad2 = np.sqrt(grad2_X**2 + grad2_Y**2)

    grad2_inv = np.where( grad2 > 0, grad2 , 1)
    grad2_inv = np.where( grad2 > 0, 1./grad2_inv , 0)
    
    normal_X = np.where( grad2 > 0, grad2_X * grad2_inv, 0)

    normal_Y = np.where( grad2 > 0, grad2_Y * grad2_inv, 0)
    
    #plt.quiver(X.T,Y.T,normal_X, normal_Y, units='xy', scale=0.5, angles=  'xy')
    #plt.imshow(PhaseField)
    #lt.show()
    dr = 0.001    
    rr0 = np.ones_like(R) * 25
    
    xx = -X
    yy = -Y  
    rt = np.sqrt(xx*xx + yy*yy) 
    #nx = xx / rt
    #ny = yy / rt    
    nx = normal_X
    ny = normal_Y

    
    rr0 = np.ones_like(R)
    for it in range(16):
        #xx = nx * (r_r0+rr0)
        #yy = ny * (r_r0+rr0)  
        xx = nx * (r_r0+rr0)
        yy = ny * (r_r0+rr0)  

        
        #plt.plot( (R-np.sqrt(xx*xx+yy*yy))[half,:] , '-')
        #plt.plot( (r_r0)[half,:] , 'o')
     #   plt.plot( normal_Y[half,:] , '-')
        #plt.show()
    
    
        f1 = np.zeros_like(R)
        for i in range(9):
            r_r0i = np.roll(np.roll(r_r0,shift=lbm.e[i,0],axis=0),shift=lbm.e[i,1],axis=1) 
            ri = np.sqrt( (lbm.e[i,0] - xx)**2 + (lbm.e[i,1] - yy)**2  )
            f1 = f1 + ( r_r0i - ( ri - rr0 ) )
    
        rr0 = rr0 + dr
        xx = nx * (r_r0+rr0)
        yy = ny * (r_r0+rr0)       
        f2 = np.zeros_like(R)
        for i in range(9):
            r_r0i = np.roll(np.roll(r_r0,shift=lbm.e[i,0],axis=0),shift=lbm.e[i,1],axis=1) 
            ri = np.sqrt( (lbm.e[i,0] - xx)**2 + (lbm.e[i,1] - yy)**2  )
            f2 = f2 + ( r_r0i - ( ri - rr0 ) )
            
        A = (f2 - f1) / dr       
        B = f2 - A * (rr0)
        temp = - B / A 
        rr0 =  temp#np.where( temp < 0, rr0 * 0.5, temp)
        
        pme = rr0
    
    
    
    pme = np.where(-(4 * PhaseField**2 - 1) < 0.1, 0, pme)
    plt.plot(pme[half,:], 'wo')
    plt.plot(pme[half,:], 'k-', lw=1)
    #plt.plot(Curvature[half,:], 'k+')
    plt.show()



        #plt.imshow(np.where(np.absolute(R - n00) < 4, rr0, 0), interpolation='nearest')
       # plt.colorbar()
       # plt.show()  
        
        

            
    
#    
#    laplace2 = PhaseField * (1./9 - 1.)
#    for i in range(1,9):
#        laplace2 = laplace2 +  np.roll(np.roll(PhaseField,shift=-lbm.e[i,0],axis=0),shift=-lbm.e[i,1],axis=1) / 9.
#    
#    
#    
#    grad2 = np.sqrt(grad2_X**2 + grad2_Y**2)[half, :]
#    
#    p2 = PhaseField[half, :]**2
#    grad_lengt = (1. - 4 * p2  ) * ww
#    curvature =   ( laplace2[half, :] - 2 * PhaseField[half, :] * (16 * p2 - 4. ) * ww**2 ) / grad_lengt
#    
#    
#    rrA = np.where(np.absolute(R - n00) < 6, rr0, 0)        
#    plt.plot( rrA[half,:] , 'o', label="Circ")
#    
#    rr1 = np.where(np.absolute(R - n00) < 6, 1./curvature, 0)        
#    plt.plot( rr1[half,:] , 'o', label="Lap")
#    plt.legend()
#    plt.show()    
#    
#    
    
    
    #plt.plot(laplace2[half, n00-dn:n00+dn] , c+'o')
    #plt.plot(laplace(R,n00, ww)[half, n00-dn:n00+dn], c+'-')
    #plt.plot(laplace(R,n00, ww)[half, n00-dn:n00+dn] - laplace2[half, n00-dn:n00+dn] , c+'-')
    #plt.plot(laplace2[half, n00-dn:n00+dn], c+'o')
    #plt.plot( R[half, n00-dn:n00+dn], Curvature[half, n00-dn:n00+dn], c+'o')
    #plt.plot( R[half, n00-dn:n00+dn], np.ones_like(curvature)[n00-dn:n00+dn] /  R[half, n00-dn:n00+dn] )
    #plt.plot((n00,n00), (0, 1./n00))
#    plt.figure()
#    
#    plt.plot( curvature[n00-dn:n00+dn], 'o-' )
#    plt.twinx()
#    plt.plot(phase(R,n00, ww)[half, n00-dn:n00+dn], 'k')
#    #plt.plot( np.ones_like(curvature)[n00-dn:n00+dn] *  R[half, n00-dn:n00+dn] )
#    
#    
#    
#    plt.figure()
#    plt.plot(grad2, 'o')
#    plt.plot(grad_lengt)
#    
#    #plt.plot(grad_lengt )
#    
#    #dn = 10
#    #plt.plot( curvature[n00-dn:n00+dn] )
#    #plt.plot( np.ones_like(curvature)[n00-dn:n00+dn] *  R[half, n00-dn:n00+dn] )
#    ##plt.plot(grad_lengt, 'o-')
#    
#    plt.figure()
#    plt.plot(phase(R,n00, ww)[half, :])
#    plt.plot(PhaseField[half,:], 'o')
    
    
plt.show()
