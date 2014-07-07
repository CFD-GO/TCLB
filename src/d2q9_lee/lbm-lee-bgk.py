# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 17:55:48 2013

@author: michal
"""


import scipy.interpolate as inter
import numpy as np
import matplotlib.pyplot as plt
import numpy.random
import os

os.system('rm /tmp/fig*')


import vtk
import vtk.util.numpy_support as VN

#G = np.zeros(9)
#G[1:5] = 1.
#G[5:] = 1./4.

#print G

#A = -0.152
NX = 64
NY = NX
nt = 3
pt = 1

plotMe = True
tau = 1.


cs2 = 1./3.
dt = 1.
dx = 1.

#def PR(rho):

    #return c2*rho*(-b2**3*rho**3/64+b2**2*rho**2/16+b2*rho/4+1)*T/(1-b2*rho/4)**3-a2*rho**2
    
#def neighbors(arr,x,y,n=3):
#    ''' Given a 2D-array, returns an nxn array whose "center" element is arr[x,y]'''
#    arr=np.roll(np.roll(arr,shift=-x+1,axis=0),shift=-y+1,axis=1)
#    return arr[:n,:n]


    




f_in = np.ones((NX,NY,9))
f_out = np.ones((NX,NY,9))

X,Y = np.meshgrid(np.arange(NX),np.arange(NY))

X = np.array(X.T,dtype=float)
Y = np.array(Y.T,dtype=float)


#rho = np.zeros_like(X)
#rho = np.sin(X / NX * 2. * np.pi) * np.sin(Y / NY * 2. * np.pi)
#ChemPot = np.sin(X / NX * 2. * np.pi) * np.sin(Y / NY * 2. * np.pi)

#gradX
#WbX = np.cos(X / NX * 2 * np.pi) / NX * 2. * np.pi * np.sin(Y / NY * 2. * np.pi)

#gradY
#WbY = np.cos(Y / NY * 2 * np.pi) / NY * 2. * np.pi * np.sin(X / NX * 2. * np.pi)

#Wb = Wb*cs2 - rho * Wb
#Laplace
#Wb = -np.sin(X / NX * 2 * np.pi) * (1. / NX * 2. * np.pi)**2



U = np.ones((NX,NY,2))


W = np.ones(9)

W[0] = 4./9.
W[1:5] = 1./9.
W[5:] = 1./36.

e = np.ndarray((2,9),dtype='int64')
e[0] = ( 0, 1, 0, -1, 0, 1, -1, -1, 1)
e[1] = ( 0, 0, 1, 0, -1, 1, 1, -1, -1)
e = e.T


X1 = X/NX 
Y1 = Y/NY 
#rho[:,:] = 1.1 - ((X1)**2 + (Y1)**2) / 2.  
#rho = 1. + np.random.random(rho.shape) / 10.

#for i in range(0,NX):
#    # 1 - 1/(1+exp(x))
h =20000.
R = np.sqrt( (X/NX -0.5)**2 + (Y/NY-0.5)**2 )
#    #rl = 1. + 0.1 * ( (1. - 1./(1.+np.exp(h * ((1.*i)/NX - 0.25)))) - (1. - 1./(1.+np.exp(h * ((1.*i)/NX - 0.75)))))
#rho =  0.2 + (1./(1.+np.exp(h * (R - 0.25)))) * 0.6
rho = np.where(X<32, np.ones_like(R)*0.8,np.ones_like(R)*0.2)
beta = 0.01
rhos_v = 0.1
rhos_l = 1.
D = 4.
kappa = beta * D**2 *(rhos_l-rhos_v)**2 / 8.
print kappa

#rho = (rhos_v+rhos_l)/2. + (np.sin(np.pi*2.*X/NX) * np.sin(np.pi*2.*Y/NY))*0.05;
#    
#    rho[:,i] = rl

#plt.imshow(rho)
#plt.show()



#rl = 0.5
#rho = rl  +  0.1 * (np.random.random_sample(rho.shape)-0.5) * 2.


#rho = 1
#rho[:,0:NY/2] = 0.2


#r = np.linspace(0.01,   3)
#E0 = beta*(r-rhos_v)**2*(r-rhos_l)**2
#ChemPot0 = 2.*beta*(r-rhos_l)**2*(r-rhos_v)+2.*beta*(r-rhos_l)*(r-rhos_v)**2
#p0 = r*ChemPot0 - E0
#plt.plot(r, p0)
#plt.show()


for i in range(0,9):
    f_in[:,:,i] = rho[:,:] * W[i]   
    f_out[:,:,i] = f_in[:,:,i]

cu = np.zeros((NX,NY,2))

hist = list()
figid = 0


def gettt(i,j,scal):
    return np.roll(np.roll(scal[:,:],shift=-i,axis=0),shift=-j,axis=1)
    
for it in range(0,nt):

    #stream

    for i in range(0,9):
        f_in[:,:,i] = np.roll(np.roll(f_out[:,:,i],shift=e[i][0],axis=0),shift=e[i][1],axis=1)

    rho[:,:] = np.sum( f_in[:,:,:], 2 )
   
    U[:,:] = 0.
    for i in range(1,9):
     U[:,:,0] =  U[:,:,0] + e[i][0]*f_in[:,:,i]  
     U[:,:,1] =  U[:,:,1] + e[i][1]*f_in[:,:,i]
    
#    U[:,:,0] = U[:,:,0] / rho[:,:]
#    U[:,:,1] = U[:,:,1] / rho[:,:]
    
    
    #F = np.zeros(U.shape)
    #d = 1
    #F[:,:,d] = 0.0001
    #F[:,:,1] = 0.
    

    #E0 = beta*(rho-rhos_v)**2*(rho-rhos_l)**2
    #ChemPot0 = 2.*beta*(rho-rhos_l)**2*(rho-rhos_v)+2.*beta*(rho-rhos_l)*(rho-rhos_v)**2
    ChemPot0 = 2.*beta*(rho-rhos_l)*(rho-rhos_v)*(2.*rho-rhos_v-rhos_l);
    #p0 = rho*ChemPot0 - E0


    LaplaceRho = np.zeros_like(rho)
    GradRhoC = np.zeros_like(U)
    GradRhoB = np.zeros_like(U)
    ChemPot = np.zeros_like(rho)    
    
    GradChemPotC = np.zeros_like(U)
    GradChemPotB = np.zeros_like(U)




        
        
    for i in range(1,9):
        scal = rho
                 
        shift = e[i]
        LaplaceRho = LaplaceRho + W[i]*(np.roll(np.roll(scal[:,:],shift=-shift[0],axis=0),shift=-shift[1],axis=1)) / cs2 / dt**2

        shift = 0.
        LaplaceRho = LaplaceRho - 2. *  W[i] * scal[:,:] / cs2 / dt**2

        shift = -e[i]
        LaplaceRho = LaplaceRho + W[i]*(np.roll(np.roll(scal[:,:],shift=-shift[0],axis=0),shift=-shift[1],axis=1)) / cs2 / dt**2


        
        shift = e[i]
        wh = 1.
        GradRhoC[:,:,0] = GradRhoC[:,:,0] + wh * W[i]*(np.roll(np.roll(scal[:,:],shift=-shift[0],axis=0),shift=-shift[1],axis=1)) / cs2 / dt**2 / 2. * e[i][0]
        GradRhoC[:,:,1] = GradRhoC[:,:,1] + wh * W[i]*(np.roll(np.roll(scal[:,:],shift=-shift[0],axis=0),shift=-shift[1],axis=1)) / cs2 / dt**2 / 2. * e[i][1]
        
        shift = -e[i]
        wh = -1.
        GradRhoC[:,:,0] = GradRhoC[:,:,0] + wh * W[i]*(np.roll(np.roll(scal[:,:],shift=-shift[0],axis=0),shift=-shift[1],axis=1)) / cs2 / dt**2 / 2. * e[i][0]
        GradRhoC[:,:,1] = GradRhoC[:,:,1] + wh * W[i]*(np.roll(np.roll(scal[:,:],shift=-shift[0],axis=0),shift=-shift[1],axis=1)) / cs2 / dt**2 / 2. * e[i][1]
        
        


        shift = e[i]
        wh = 4.
        GradRhoB[:,:,0] = GradRhoB[:,:,0] + wh * W[i]*(np.roll(np.roll(scal[:,:],shift=-shift[0],axis=0),shift=-shift[1],axis=1)) / cs2 / dt**2 / 2. * e[i][0]
        GradRhoB[:,:,1] = GradRhoB[:,:,1] + wh * W[i]*(np.roll(np.roll(scal[:,:],shift=-shift[0],axis=0),shift=-shift[1],axis=1)) / cs2 / dt**2 / 2. * e[i][1]
        
        shift = 2*e[i]
        wh = -1.
        GradRhoB[:,:,0] = GradRhoB[:,:,0] + wh * W[i]*(np.roll(np.roll(scal[:,:],shift=-shift[0],axis=0),shift=-shift[1],axis=1)) / cs2 / dt**2 / 2. * e[i][0]
        GradRhoB[:,:,1] = GradRhoB[:,:,1] + wh * W[i]*(np.roll(np.roll(scal[:,:],shift=-shift[0],axis=0),shift=-shift[1],axis=1)) / cs2 / dt**2 / 2. * e[i][1]

        shift = 0*e[i]
        wh = -3.
        GradRhoB[:,:,0] = GradRhoB[:,:,0] + wh * W[i]*(np.roll(np.roll(scal[:,:],shift=-shift[0],axis=0),shift=-shift[1],axis=1)) / cs2 / dt**2 / 2. * e[i][0]
        GradRhoB[:,:,1] = GradRhoB[:,:,1] + wh * W[i]*(np.roll(np.roll(scal[:,:],shift=-shift[0],axis=0),shift=-shift[1],axis=1)) / cs2 / dt**2 / 2. * e[i][1]
        
   
    ChemPot = ChemPot0 - kappa*LaplaceRho

    

    for i in range(1,9):
        scal = ChemPot      

        shift = e[i]
        wh = 1.
        GradChemPotC[:,:,0] = GradChemPotC[:,:,0] + wh * W[i]*(np.roll(np.roll(scal[:,:],shift=-shift[0],axis=0),shift=-shift[1],axis=1)) / cs2 / dt**2 / 2. * e[i][0]
        GradChemPotC[:,:,1] = GradChemPotC[:,:,1] + wh * W[i]*(np.roll(np.roll(scal[:,:],shift=-shift[0],axis=0),shift=-shift[1],axis=1)) / cs2 / dt**2 / 2. * e[i][1]
        
        shift = -e[i]
        wh = -1.
        GradChemPotC[:,:,0] = GradChemPotC[:,:,0] + wh * W[i]*(np.roll(np.roll(scal[:,:],shift=-shift[0],axis=0),shift=-shift[1],axis=1)) / cs2 / dt**2 / 2. * e[i][0]
        GradChemPotC[:,:,1] = GradChemPotC[:,:,1] + wh * W[i]*(np.roll(np.roll(scal[:,:],shift=-shift[0],axis=0),shift=-shift[1],axis=1)) / cs2 / dt**2 / 2. * e[i][1]


        shift = e[i]
        wh = 4.
        GradChemPotB[:,:,0] = GradChemPotB[:,:,0] + wh * W[i]*(np.roll(np.roll(scal[:,:],shift=-shift[0],axis=0),shift=-shift[1],axis=1)) / cs2 / dt**2 / 2. * e[i][0]
        GradChemPotB[:,:,1] = GradChemPotB[:,:,1] + wh * W[i]*(np.roll(np.roll(scal[:,:],shift=-shift[0],axis=0),shift=-shift[1],axis=1)) / cs2 / dt**2 / 2. * e[i][1]
        
        shift = 2*e[i]
        wh = -1.
        GradChemPotB[:,:,0] = GradChemPotB[:,:,0] + wh * W[i]*(np.roll(np.roll(scal[:,:],shift=-shift[0],axis=0),shift=-shift[1],axis=1)) / cs2 / dt**2 / 2. * e[i][0]
        GradChemPotB[:,:,1] = GradChemPotB[:,:,1] + wh * W[i]*(np.roll(np.roll(scal[:,:],shift=-shift[0],axis=0),shift=-shift[1],axis=1)) / cs2 / dt**2 / 2. * e[i][1]

        shift = 0*e[i]
        wh = -3.
        GradChemPotB[:,:,0] = GradChemPotB[:,:,0] + wh * W[i]*(np.roll(np.roll(scal[:,:],shift=-shift[0],axis=0),shift=-shift[1],axis=1)) / cs2 / dt**2 / 2. * e[i][0]
        GradChemPotB[:,:,1] = GradChemPotB[:,:,1] + wh * W[i]*(np.roll(np.roll(scal[:,:],shift=-shift[0],axis=0),shift=-shift[1],axis=1)) / cs2 / dt**2 / 2. * e[i][1]

#    gradients()
#    VV = GradRhoB
#    plt.plot(VV[1,:,1])
#    plt.plot(VV[1,:,0])
#    plt.plot(VV[:,1,1])
#    plt.plot(VV[:,1,0])
#
#    VV = GradRhoC
#    plt.plot(VV[1,:,1], 'o')
#    plt.plot(VV[1,:,0], 'o')
#    plt.plot(VV[:,1,1], 'o')
#    plt.plot(VV[:,1,0], 'o')
#    
#    
#    plt.plot(WbX[1,:],'--',lw=3)
#    plt.plot(WbX[:,1],'--',lw=3)
#    
#    plt.figure()
#    plt.contourf(X,Y,rho)
#    plt.quiver(X,Y,VV[:,:,0],VV[:,:,1])
#    plt.show()
    #GradChemPotM = 0.5 * ( GradChemPotB + GradChemPotC )
    #GradRhoM = 0.5 * ( GradRhoB + GradRhoC )

    GradC = np.zeros_like(GradRhoC)
    GradB = np.zeros_like(GradRhoB)
    
    GradC[:,:,0] = GradRhoC[:,:,0] * cs2 - rho * GradChemPotC[:,:,0]
    GradC[:,:,1] = GradRhoC[:,:,1] * cs2 - rho * GradChemPotC[:,:,1]
    GradB[:,:,0] = GradRhoB[:,:,0] * cs2 - rho * GradChemPotB[:,:,0]
    GradB[:,:,1] = GradRhoB[:,:,1] * cs2 - rho * GradChemPotB[:,:,1]
    

    
    
    GradM = 0.5  * ( GradC + GradB )
    #F[1:-1,:,0] = F[1:-1,:,0] + 0.000001*rho[1:-1,:]
    
    #F[0,:,:] = F[0,:,:] - U[0,:,:] 
    #F[NX-1,:,:] = F[NX-1,:,:] - U[NX-1,:,:] 
  #  F[:,:,1] = F[:,:,1] / rho
  #  F[:,:,0] = F[:,:,0 ] / rho
    
    
    U[:,:,0] = U[:,:,0] + dt * 0.5 * ( GradC[:,:,0] ) 
    U[:,:,1] = U[:,:,1] + dt * 0.5 * ( GradC[:,:,1] ) 
    
    U[:,:,0] = U[:,:,0] / rho[:,:]
    U[:,:,1] = U[:,:,1] / rho[:,:]
    
    #collide    
    
    #F[:,:,1] = GradRhoC[:,:,1] * cs2 - rho * GradChemPotC[:,:,1]
    #F[:,:,0] = GradRhoC[:,:,0] * cs2 - rho * GradChemPotC[:,:,0]
    #F[:,:,0] = GradM[:,:,0] 
    #F[:,:,1] = GradM[:,:,1] 
    
    #Test = np.zeros_like(U)
    #Test = np.zeros_like(rho)
    
    Grad_Cumm = np.zeros_like(U)
    feq0 = np.zeros_like(f_in)
    fB = np.zeros_like(f_in)
    def compute(i):
    #for i in range(9):
     
            cu  = 3. * ( U[:,:,0] * e[i,0] + U[:,:,1] * e[i,1])
            feq0[:,:,i] = W[i] * rho[:,:] * (1. + cu[:,:] + 0.5*cu[:,:]*cu[:,:] - (3./2.) * ( U[:,:,0]**2 + U[:,:,1]**2 ) )
    
                

            GradC_Directional = np.zeros_like(rho)
            GradB_Directional = np.zeros_like(rho)


            #GradC_Directional2 = np.zeros_like(rho)
            #GradB_Directional2 = np.zeros_like(rho)
            
     #       if i > 0: 

            sub = 2. #* 3.
            
            scal = rho * cs2
            shift = e[i]
            wh = 1.        
            GradC_Directional[:,:] = wh * (np.roll(np.roll(scal[:,:],shift=-shift[0],axis=0),shift=-shift[1],axis=1)) /  sub
            shift = -e[i]
            wh = -1.        
            GradC_Directional[:,:] = \
                GradC_Directional[:,:] + wh * (np.roll(np.roll(scal[:,:],shift=-shift[0],axis=0),shift=-shift[1],axis=1)) /  sub


            scal = rho * cs2
            
            shift = 2 * e[i]
            wh = -1.        
            GradB_Directional[:,:] = \
                wh * (np.roll(np.roll(scal[:,:],shift=-shift[0],axis=0),shift=-shift[1],axis=1)) /  sub
            shift = e[i]
            wh = 4.        
            GradB_Directional[:,:] = \
                GradB_Directional[:,:] + wh * (np.roll(np.roll(scal[:,:],shift=-shift[0],axis=0),shift=-shift[1],axis=1)) /  sub
            shift = 0*e[i]
            wh = -3.        
            GradB_Directional[:,:] = \
                GradB_Directional[:,:] + wh * (np.roll(np.roll(scal[:,:],shift=-shift[0],axis=0),shift=-shift[1],axis=1)) /  sub
                
                
          
            scal = ChemPot       
            shift = e[i]
            wh = 1.        
            GradC_Directional[:,:] = \
                GradC_Directional[:,:] - rho * wh * (np.roll(np.roll(scal[:,:],shift=-shift[0],axis=0),shift=-shift[1],axis=1)) /  sub   
            shift = -e[i]
            wh = -1.        
            GradC_Directional[:,:] = \
                GradC_Directional[:,:] - rho * wh * (np.roll(np.roll(scal[:,:],shift=-shift[0],axis=0),shift=-shift[1],axis=1)) /  sub
            
            scal = ChemPot
            
            shift = 2 * e[i]
            wh = -1.        
            GradB_Directional[:,:] = \
                GradB_Directional[:,:] - rho * wh * (np.roll(np.roll(scal[:,:],shift=-shift[0],axis=0),shift=-shift[1],axis=1)) /  sub
            shift = e[i]
            wh = 4.        
            GradB_Directional[:,:] = \
                GradB_Directional[:,:] - rho * wh * (np.roll(np.roll(scal[:,:],shift=-shift[0],axis=0),shift=-shift[1],axis=1)) /  sub
            shift = e[0]
            wh = -3.        
            GradB_Directional[:,:] = \
                GradB_Directional[:,:] - rho * wh * (np.roll(np.roll(scal[:,:],shift=-shift[0],axis=0),shift=-shift[1],axis=1)) /  sub
            
            
            
            

            
            
            fB[:,:,i] = GradB_Directional[:,:]


            #plt.show()
            #GradC_Directional = GradC[:,:,0]*e[i][0] + GradC[:,:,1]*e[i][1]
            #GradB_Directional = GradB[:,:,0]*e[i][0] + GradB[:,:,1]*e[i][1]
            
            feq = feq0[:,:,i] - dt / 2. * (GradC_Directional - U[:,:,0]*GradC[:,:,0] - U[:,:,1]*GradC[:,:,1]) / rho / cs2 * feq0[:,:,i]



            GradM_Directional = 0.5 * (GradC_Directional + GradB_Directional)
            #f_out[:,:,i] = f_in[:,:,i]            
            f_out[:,:,i] = \
                f_in[:,:,i] - 1. / tau * (f_in[:,:,i] - feq) + \
                ( GradM_Directional - U[:,:,0]*GradM[:,:,0] - U[:,:,1]*GradM[:,:,1]) / rho / cs2 * feq0[:,:,i]
#                
            #f_out[:,:,i] = feq0[:,:,i] + 0.5*( GradB_Directional - U[:,:,0]*GradB[:,:,0] - U[:,:,1]*GradB[:,:,1]) / rho / cs2 * feq0[:,:,i]
                        
#            if i>0:
#                plt.figure()
#                plt.plot(f_out[:,:,i] - akuku[:,:])
#                plt.show()
                
    print "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    for j in range(9):
        compute(j)
            
            


fname = '/home/michal/tach-17/home/tjanson/projekty/TCLB/output/drop_lee_VTK_P00_0000000'+str(nt-1)+'.vti'
reader = vtk.vtkXMLImageDataReader()
reader.SetFileName(fname)
#reader.ReadAllVectorsOn()
#dareader.ReadAllScalarsOn()
reader.Update()

data = reader.GetOutput()
dim = data.GetDimensions()


s_scal = [dim[0]-1, dim[1]-1]
s_vec = [dim[0]-1, dim[1]-1,3]


rho_ll = VN.vtk_to_numpy(data.GetCellData().GetArray('Rho'))
rho_ll = rho_ll.reshape(s_scal,order='F')

U_ll = VN.vtk_to_numpy(data.GetCellData().GetArray('U'))
U_ll = U_ll.reshape(s_vec,order='F')


plt.plot(rho_ll[:,2]-rho[:,2], label='drho')
plt.legend(loc=1)
plt.twinx()
plt.plot(rho[:,2], 'k-x', label='rho')
plt.legend(loc=2)
plt.figure()


plt.plot(U_ll[:,2,0]-U[:,2,0], label='dux')
plt.plot(U_ll[:,2,1]-U[:,2,1], label='duy')
plt.legend()

plt.figure()

Fb_ll = VN.vtk_to_numpy(data.GetCellData().GetArray('FB'))
Fb_ll = Fb_ll.reshape(s_vec,order='F')

Fc_ll = VN.vtk_to_numpy(data.GetCellData().GetArray('FC'))
Fc_ll = Fc_ll.reshape(s_vec,order='F')

plt.plot(Fb_ll[:,2,0]-GradB[:,2,0], label='dFbx')
plt.plot(Fb_ll[:,2,1]-GradB[:,2,1], label='dFby')
plt.legend()

plt.figure()
plt.plot(Fc_ll[:,2,0]-GradC[:,2,0], label='dFcx')
plt.plot(Fc_ll[:,2,1]-GradC[:,2,1], label='dFcy')
plt.legend()

u = U
Fb2_llx = ( ( -fB[:,:,8] - fB[:,:,7] + fB[:,:,6] + fB[:,:,5] + ( -fB[:,:,4] + fB[:,:,2] )*4. )*0. + ( fB[:,:,8] - fB[:,:,7] - fB[:,:,6] + fB[:,:,5] + ( -fB[:,:,3] + fB[:,:,1] )*4. )*1. )/12. ;
Fb2_lly = ( ( -fB[:,:,8] - fB[:,:,7] + fB[:,:,6] + fB[:,:,5] + ( -fB[:,:,4] + fB[:,:,2] )*4. )*1. + ( fB[:,:,8] - fB[:,:,7] - fB[:,:,6] + fB[:,:,5] + ( -fB[:,:,3] + fB[:,:,1] )*4. )*0. )/12. ;

Fb1_llx = ( fB[:,:,8] - fB[:,:,7] - fB[:,:,6] + fB[:,:,5] + ( -fB[:,:,3] + fB[:,:,1] )*4. )/12. ;
Fb1_lly = ( -fB[:,:,8] - fB[:,:,7] + fB[:,:,6] + fB[:,:,5] + ( -fB[:,:,4] + fB[:,:,2] )*4. )/12. ;



GradB2 = np.zeros_like(U)

for i in range(9):
    for j in range(2):
        GradB2[:,:,j] = GradB2[:,:,j] + fB[:,:,i]*e[i,j]*W[i]  / cs2
    


test_ll = VN.vtk_to_numpy(data.GetCellData().GetArray('TEST'))
test_ll = test_ll.reshape(s_scal,order='F')

plt.figure()
plt.plot(Fb1_llx[:,2], 'x-')
plt.plot(Fb2_llx[:,2], 'x-')
plt.plot(GradB[:,2,0], 'o-')

f_ll = np.zeros_like(f_in)


fname = '/home/michal/tach-17/home/llaniewski/drop2_VTK_P00_0000000'+str(nt-1)+'.vti'
reader2 = vtk.vtkXMLImageDataReader()
reader2.SetFileName(fname)
reader2.Update()
data2 = reader2.GetOutput()
dim = data2.GetDimensions()

for i in range(9):
    temp = VN.vtk_to_numpy(data2.GetCellData().GetArray('F'+str(i)))
    temp = temp.reshape(s_scal,order='F')
    f_ll[:,:,i] = temp
    
    print i
    print np.max(f_ll[:,2,i]-f_in[:,2,i])
    
#plt.figure()
#
#plt.plot(f_ll[:,2,1:4]-f_in[:,2,1:4])
#
#    
#plt.show()
