# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 14:43:51 2015

@author: mdzikowski
"""



import matplotlib as mpl
fresult='/tmp/final.vti'
fconfig='/home/mdzikowski/tachion/home/mdzikowski/projekty/TCLB/example/npe_guo.xml'

mpl.rcParams['lines.linewidth'] = 4
mpl.rcParams['lines.markersize'] = 12
mpl.rcParams['lines.markeredgewidth'] = 2.
mpl.rcParams['lines.color'] = 'w'
font = {
        'size'   : 20}

mpl.rc('font', **font)

import matplotlib.pyplot as plt

import numpy as np
import vtk
import vtk.util.numpy_support as VN
import xml.sax
import re



class CLBXMLHandler(xml.sax.ContentHandler):
    
    def __init__(self, config_ref):
        self.config = config_ref
        
    def startElement(self, name, attrs):
        if name == "Params":
            a = dict()
            for (k,v) in attrs.items():
                if k == 'gauge':
                    a['gauge'] = float(v)
                else:
                    a['name'] = k
                    a['value'] = v
                    a['float'] = float(re.findall('[-,\.,0-9]+', v)[0])
                    
                
            self.config[a['name']] = a

class VTIFile:
    def __init__(self, vtifname, parallel=False):
        self.reader = vtk.vtkXMLImageDataReader()
        self.reader.SetFileName(vtifname)
        self.reader.Update()
        self.data = self.reader.GetOutput()  
        self.dim = self.data.GetDimensions()   
        self.s_scal = [self.dim[1]-1, self.dim[0]-1]
        self.s_vec = [self.dim[1]-1, self.dim[0]-1,3]

    def get(self, name, vector=False):
        if vector:
            return VN.vtk_to_numpy(self.data.GetCellData().GetArray(name)).reshape(self.s_vec)
        else:
            return VN.vtk_to_numpy(self.data.GetCellData().GetArray(name)).reshape(self.s_scal)
            
CLBc = dict()
parser = xml.sax.make_parser()
parser.setContentHandler(CLBXMLHandler(CLBc))
parser.parse(open(fconfig,"r"))


CLBcf = dict()
CLBcg = dict()
for c in  CLBc:
    CLBcf[c] = CLBc[c]['float']
    if 'gauge' in CLBc[c]:
        CLBcg[c] = CLBc[c]['gauge']
        print c, "  gauge = ", CLBc[c]['gauge']        
    print c, " = ", CLBc[c]['float']
    

    
An = 6.022*100000000000000000000000.

    
n_inf = CLBcf['n_inf'] * An  # na m^3
n_inf_g = n_inf * (CLBcf['x'] / CLBcg['x']) ** 3
print "n_inf_g = ", n_inf_g
el_kbT = CLBcf['el_kbT']
#T = CLBcf['T']
z = CLBcf['ez']
el = CLBcf['el']
epsilon = CLBcf['epsilon']
zeta = CLBcf['psi_bc-wall']
E0 = CLBcf['Ex']
mu = CLBcf['nu'] 

z0 = z
z1 = -z

lambda_d = np.sqrt( epsilon / el_kbT / 2. / n_inf / z**2 / el  )

#lambda_d_g = np.sqrt( epsilon_g * kb_g * T_g / 2. / n_inf_g / z**2 / el_g**2  )

kappa = CLBcf['x'] / lambda_d

print "kappa = ", CLBcf['x'] / lambda_d
#print "kappa_g =",  (CLBcg['x']) / lambda_d_g




#################################3
# Read VTI
#################################



VTIf =  VTIFile(fresult)

Ny = VTIf.dim[1]

psi_lb = VTIf.get('Psi').T
n0_ln = VTIf.get('n0').T
n1_lb = VTIf.get('n1').T

U_lb = VTIf.get('U', True)
F_lb = VTIf.get('F', True)
gradPsi_lb = VTIf.get('GradPsi', True)



#plt.show()


 
ex = np.exp
dy = 0.5 / Ny
yy = np.linspace(dy,1.-dy,Ny-3)

psi00 = (ex(kappa) - 1.) / (ex(kappa)-ex(-kappa)) * ex(-kappa*yy) + (1. - ex(-kappa)) / (ex(kappa)-ex(-kappa)) * ex(kappa*yy)

gradPsi00 = -kappa*(ex(kappa) - 1.) / (ex(kappa)-ex(-kappa)) * ex(-kappa*yy) + kappa*(1. - ex(-kappa)) / (ex(kappa)-ex(-kappa)) * ex(kappa*yy)
##print psi00
#
##plt.plot(yy,zeta*psi00)
##plt.show()
#
#n00 = n_inf * np.exp(-z0 * el * psi00 / kb / T)
#n10 = n_inf * np.exp(-z1 * el * psi00 / kb / T)

U00 = - epsilon * zeta * E0 / mu * (1. -  (ex(kappa*yy)+ex(kappa-kappa*yy))/(1.+ex(kappa)))

plt.plot(yy,U_lb[1:-1,0,0])
#plt.twinx()
plt.plot(yy,U00, 'ko')

#plt.figure()
#plt.plot(yy,gradPsi_lb[1:-1,0,1])
#plt.twinx()
#plt.plot(yy,gradPsi00,'k')
#plt.plot(U00/U_lb[:,0,0])
plt.show()

