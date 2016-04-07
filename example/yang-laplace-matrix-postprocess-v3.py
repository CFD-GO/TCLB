# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 20:26:07 2015

@author: michal
"""

import bearded_octo_wookie.CLB.CLBXMLHandler as CLBXML
from bearded_octo_wookie.CLB import *
from bearded_octo_wookie import *
import os
import glob 
import numpy as np
import matplotlib.pyplot as plt
import bearded_octo_wookie.jednostki2 as j2
import scipy.optimize as so
import re


import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.patches as mp
norm = mpl.colors.Normalize(vmin=0, vmax=20)
cmap = cm.gray
#x = 0.3

scmap = cm.ScalarMappable(norm=norm, cmap=cmap)


fnames = list()

fname_base = './example/tmp/matrix'
   
#for R in [256]: 
#    for rhoW in [0.5, 1]:
#        H = 2 * R
#        for LdoR in [0.2, 0.3, 0.4, 0.5]:
#            for hdoR in [0.05, 0.1, 0.15, 0.2]:            
#                #createConfig(R,H,LdoR,hdoR,rhoW, '/home/michal/tach-17/mnt/fhgfs/users/mdzikowski/yang-laplace-sphere-matrix/matrix-')
#                fconfig = fname_base+"rhoW_" + str(rhoW) +"R_" + str(R) + "H_"+str(H) + "LdoR_"+str(LdoR) + "hdoR_"+str(hdoR) 
#                fnames.append(fconfig)
#

print "Collecting files from, ", fname_base
fnames = glob.glob(fname_base+'*.xml')
print "Data collected..."
#print fnames
#==============================================================================
# MatrixData = np.zeros(1, dtype=('i4',[
#     ('BodyForce','f4'), 
#     ('pressureForce','f4'), 
#     ('R','i4'), 
#     ('LdoR','f4'), 
#     ('hdoR','f4'), 
#     ('rhoW','f4')    
#     ]))
#==============================================================================

def getData(fnames, **kwargs):
    MatrixData = list()
    i = 0
    j = 0
    
    if kwargs.has_key('plots'):    
        rhoWs = kwargs['rhoWs']
    
        plts = {}
        
        for r in rhoWs:
            fig = plt.figure()
            plts[r] = fig.add_subplot(111)
            plts[r].grid(which='both')


    for fconfig in fnames:
            fconfig = fconfig[:-4]
            #print fconfig
            print i+1, '/', len(fnames)
            i = i + 1
  #          try:
            CLBc, CLBcf, CLBcg = CLBXML.parseConfig(fconfig+'.xml', multiparams=True)
                
            k = CLBcf['Magic']
            u = CLBcf['MovingWallVelocity']
            g = -CLBcf['GravitationX']
            T = CLBcf['Temperature']
            r = CLBcf['Density']
            RhoBB = CLBcf['Density-zwall']
            
            omega = 1 +  CLBcf['S2']
            mu = (omega - 0.5)/3. * r         


            ###########################################
            ############## READ VTI ###################
            ###########################################
            fconfig = fconfig[14:]
            flist = glob.glob('./output/'+fconfig+'_VTK*.vti') 
            if len(flist) > 0:

                    
                fvti = flist[0]
                config = dict()            
                for var in ("rhoW","R","H","LdoR","hdoR"):
                    config[var] = float(re.match('.*[\-0-9]'+var+'_([0-9.]*).*', fvti).group(1))
                    
                vti = VTIFile.VTIFile(fvti, False)
                vti.trim(x0=1, x1=-2)
                
                F = vti.get('F', vector=True)
                Rho = vti.get('Rho', vector=False)
                Boundary = vti.get('BOUNDARY', vector=False)
                Collision = vti.get('COLLISION', vector=False)
                P = vti.get('P', vector=False)
                #P = k *  j2.PR(Rho, 0.56)
                
                #Fabs = np.sqrt(F[:,:,0]*F[:,:,0]+F[:,:,1]*F[:,:,1])
                Solid = Boundary == 1
                Fluid = Collision == 2
                
                xsym = np.argmax(Boundary[1,:]) - 1
                ysym = int(len(Boundary[:,0]) / 2)
                
                X,Y = vti.getMeshGrid()
                ############################################
                ############ COMPUTE FORCES ################
                ############################################
        
                psi0 = np.sqrt(-k*j2.PR(RhoBB, T) + RhoBB/3.)
                L_BB =  float(len(Rho[:,-1]))
                #F[Solid] - force acting on the SOLID, computed with Ladd
                F_bb0 = np.sum(-F[Solid][:,0])
                F_bb = - (F_bb0 + L_BB * (RhoBB / 3. - psi0**2 )  )
                
    
    
                #plt.plot(Rho[:,xsym])
                xs = Y[:,xsym]
                Lx = np.max(xs)
                xs = xs / float(Lx)
                Rho_xs = Rho[:,xsym]
                rhoMax = np.max(Rho_xs)
                rhoMin = np.min(Rho_xs)    
                
                
                coeff0 = [0.2, 0.6, 400]
                
                def getInterpolator(coeff):
                    x0, x1, h = coeff
                    return lambda(x): rhoMin + (rhoMax-rhoMin)* (
                        1./(1.+np.exp(h * (x-x1)))
                        - 1./(1.+np.exp(h * (x-x0)))
                        )
                    
                res, cc=  so.leastsq(
                    lambda(coeff): Rho_xs - getInterpolator(coeff)(xs), 
                    coeff0,
                    full_output=0
                )                
    
                x0, x1, h = res
                L = (x1 - x0) * float(Lx)
                f = getInterpolator(res)
                
                #plt.plot(xs,f(xs))
                #plt.plot(xs,Rho[:,xsym],'o')            
    
    
                P0 = P[0,xsym]
                P1 = P[ysym,xsym]
                
                dP = P0- P1
                sigma = j2.surTension_v2(T, k)
                
                #print 'P0, P1 =', P0, P1
                #print 'dP=',dP
                #print 'L*dP=',dP*L
                #print '2Sigma', 2 * sigma
                #print 'Fcap=', F_cap
                #print 'Fbb=',F_bb
                #print 'sum=',F_bb - F_cap
                #print 'L_num= ', (F_bb - sigma) / dP
                #print 'L', L
                #print 'Sigma_num', (F_bb - dP*L) 
                #print 'Sigma', sigma




                if kwargs.has_key('plots'):
                    
                    #C = np.where(Rho < 0.99, 0, Rho)
                    rw = config['rhoW']
                    cplt = plts[rw]
                    LdoR = config['LdoR']
                    R = config['R']
                    hdoR = config['hdoR']
                    
                    h0 =  int( hdoR * R )
                    L0 =  int( LdoR * R) 
                    print h0*L0
                    dX = 2 * R - L0 - 1
                    
                    Rho1 = np.where(X > xsym, 0, Rho)
                    X = X + dX
                        
                    if LdoR <= kwargs['L_break'][rw] and LdoR>0.03:
                        cplt.axis('equal')
                        color = scmap.to_rgba(j)[:-1]
                        cplt.contour(X,Y,Rho1, colors=[color], levels=[1], lw = 2)
                        #RR = np.where(X**2 + (Y - np.max(Y) / 2. )**2 < (2.*R)**2, 1, 0  )
                        RR = np.sqrt((X - R )**2 + (Y - np.max(Y) / 2.)**2)
                        #RR = np.where(RR <= R**2, RR ,0)
                        
                        cplt.contour(X,Y,RR, levels=(R-2,R-1,R), colors=('w', 'k', 'k') )
                        #fig = plt.figure()
                        #circle = mp.Circle((X-R, Y - np.max(Y) / 2., ), R)
                        #circle2 = mp.Circle((0.,0.), 0.1)                        
                        #fig.gca().add_artist(circle)
                        #fig.gca().add_artist(circle2)
                        #cplt.gca()
                        #plt.show()
                        
                    #plt.plot((xsym + dX, xsym + dX), (0,np.max(Y)), c=color)                
                    #plt.show()
                    
                #internal force            
                    #
                    #plt.plot(xs,P[:,xsym],'o')            
                    #print x0, x1
                    #plt.imshow(Rho)
                    #plt.show()            
                    #plt.contourf(X,Y,Fabs*Solid, interpolation='nearest')
                    #plt.show()
                MatrixData.append([
                      config['R'], #0
                      config['rhoW'], #1
                      config['LdoR'], #2
                      config['hdoR'], #3
                      F_bb, #4
                      dP, #5
                      sigma, #6
                      L, #7
                      h #8
                ])
               # np.savez(fname_base+'cache.npz', md=MatrixData) 
               # except Exception:
               #     print fconfig
    ##########################################
    MatrixData = np.array(MatrixData)
    return MatrixData
    
plt.show() 
#try:
#    mtime = os.stat(fname_base+'cache.npz').st_mtime 
#except OSError:
#    mtime = 0   
#
#if os.stat(fnames[0]).st_mtime >  mtime :
if True:
    MatrixData = getData(fnames)
    np.savez(fname_base+'-cache.npz', md=MatrixData)    
else:
    print "CACHED"
    MatrixData = np.load(fname_base+'-cache.npz')['md']

MatrixData = np.load(fname_base+'-cache.npz')['md']
    
rhoWs = np.unique(MatrixData[:,1])
LdoRs = np.unique(MatrixData[:,2])
hdoRs = np.unique(MatrixData[:,3])

print "Detected rhoWs:", rhoWs
print "Detected LdoRs", LdoRs
print "Detected hdoRs", hdoRs
#force, iso volume

for rhoW in rhoWs:    
    plt.figure()
    plt.title(r'$\rho=$'+str(rhoW))
    plt.grid(which='both')
#    for hdoR in hdoRs:
#        hdoR_Idx = np.extract(MatrixData[:,3] == hdoR, np.arange(len(MatrixData[:,3])))
#        hdoR_Data = MatrixData[hdoR_Idx,:]    
 #       rhoW_Idx = np.extract(hdoR_Data[:,1] == rhoW, np.arange(len(hdoR_Data[:,3])))
    rhoW_Idx = np.extract(MatrixData[:,1] == rhoW, np.arange(len(MatrixData[:,3])))    
    Data = MatrixData[rhoW_Idx,:]   
    IdxSorted = np.argsort(Data[:,2])
    Data = Data[IdxSorted,:]
        
    plt.plot(Data[:,2], Data[:,4], 'o-')
   # plt.legend(loc=0)


L_break = {
1.5: 0.2,
0.5: 0.12 
}

getData(fnames, plots=True, rhoWs=rhoWs, L_break=L_break)



plt.figure()
Pcap = MatrixData[:, 5] * MatrixData[:, 7]  + MatrixData[:, 6] 
print Pcap
plt.plot(MatrixData[:, 4], 'o')
plt.plot(Pcap, 'o')
plt.show()