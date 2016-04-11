# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 20:26:07 2015

@author: michal
"""

import bearded_octo_wookie.CLB.CLBXMLWriter as CLBXML



def createConfig( R, H, LdoR, hdoR, rhoW, prefix ):
    NX = int( ( 2 *  LdoR ) * R )
    NY = int( H )
    h =  int( hdoR * R )
    L =  int( LdoR * R) 
    
    #print L
    
    
    CLBc = CLBXML.CLBConfigWriter( \
    " R=" + str(R) +
    " RhoW=" + str(rhoW) +
    " H="+str(H) +
    " LdoR="+str(LdoR) +
    " hdoR="+str(hdoR) 
    )
    fname = prefix+"rhoW_" + str(rhoW) +"R_" + str(R) + "H_"+str(H) + "LdoR_"+str(LdoR) + "hdoR_"+str(hdoR) 
    CLBc.addGeomParam('nx', NX)
    CLBc.addGeomParam('ny', NY)
    
    
    
    CLBc.addMRT()
    CLBc.addBox()
    
    CLBc.addZoneBlock(name='zwet')
    CLBc.addBox(dy=int(NY/2 - h), fy=int(NY/2 + h))
    
    CLBc.addWall(name="zwall")
    #CLBc.addSphere(dy=">128", ny="256", dx=">-128", nx="256")
    sphere = CLBc.addSphere(dy="0", ny=2*R, dx=">-"+str(2*R-L-1), nx=2*R)
    CLBc.addBox(fy=-1, nx=int(0.5*L))
    
    CLBc.addRightSymmetry()
    CLBc.addBox(fy=-1, dx=-1)
    
    #CLBc.addTopSymmetry()
    #CLBc.addBox(fx=-1, dy=-1)
    
    params = {
    
    'Density':"0.05",
    'Density-zwet':"3.117355002492964819",
    'Density-zwall':rhoW,
    'Magic':"0.008",
    
    'Temperature':"0.56",
    'GravitationY':"-0.00000",
    'GravitationX':"-0.00000",
    
    'FAcc':"1",
    'MagicA':"-0.152",
    'MagicF':"-0.6666666666666",
    'MovingWallVelocity':"0.000",
    'InletVelocity': "0.0",
    
    'S0':"0",
    'S1':"0",
    'S2':"0",
    'S3':"-0.333333",
    'S4':"0",
    'S5':"0",
    'S6':"0",
    'S7':"0.00",
    'S8':"0.00"
    
    }
    
    for n in params:
        CLBc.addModelParam(n, params[n])
        
    CLBc.addSolve(iterations=1, vtk=1)     
    CLBc.addSolve(iterations=1000 * NX, vtk=250 * NX) 
    
    #CLBc.dump()
    #f = file('/tmp/list.txt', 'a')
    print fname+'_VTK_P00_%.8d.vti'%int(1000*NX+1)
    CLBc.write(fname+'.xml')
    #for l in file('/tmp/a.xml'):
    #    print l
    
    
########################################################3
    
    
#==============================================================================
# for R in [128, 256, 512, 1024]: 
#     for rhoW in [0.5, 1 , 3]:
#         H = 2 * R
#         for LdoR in [0.2, 0.3, 0.4, 0.5]:
#             for hdoR in [0.05, 0.1, 0.15, 0.2]:            
#                 #createConfig(R,H,LdoR,hdoR,rhoW, '/home/michal/tach-17/mnt/fhgfs/users/mdzikowski/yang-laplace-sphere-matrix/matrix-')
#                 createConfig(R,H,LdoR,hdoR,rhoW, '/tmp/y/matrix-')
# 
#==============================================================================
##########################################


#==============================================================================
# for R in [256, 512]: 
#  for rhoW in [0.5, 1 , 1.5]:
#      H = 2 * R
#      for LdoR in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.,7, 0.8, 0.9, 1.]:
#          for hdoR in [0.05, 0.1, 0.15, 0.2]:            
#              #createConfig(R,H,LdoR,hdoR,rhoW, '/home/michal/tach-17/mnt/fhgfs/users/mdzikowski/yang-laplace-sphere-matrix/matrix-')
#              createConfig(R,H,LdoR,hdoR,rhoW, '/tmp/y/matrix2-')
#      
#==============================================================================


for R in [256]: 
  for rhoW in [0.5, 1.5]:
      H = 2 * R
      #print "XXXXXXXXXXXXXXXXXXXXXXXXXx"
      for LdoR in range(1,30):
          LdoR = float(LdoR) / 30. * 0.6
          #for V in [0.0025]:
          for V in [0.005]:
              hdoR = V / LdoR
       #       print hdoR
              createConfig(R,H,LdoR,hdoR,rhoW, '/tmp/y/matrix3.2-')
          #for hdoR in [0.05, 0.2]:            

              #createConfig(R,H,LdoR,hdoR,rhoW, '/home/michal/tach-17/mnt/fhgfs/users/mdzikowski/yang-laplace-sphere-matrix/matrix-')
