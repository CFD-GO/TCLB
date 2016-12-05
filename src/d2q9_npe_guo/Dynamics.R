
AddDensity( name="phi[0]", dx= 0, dy= 0, group="phi")
AddDensity( name="phi[1]", dx= 1, dy= 0, group="phi")
AddDensity( name="phi[2]", dx= 0, dy= 1, group="phi")
AddDensity( name="phi[3]", dx=-1, dy= 0, group="phi")
AddDensity( name="phi[4]", dx= 0, dy=-1, group="phi")
AddDensity( name="phi[5]", dx= 1, dy= 1, group="phi")
AddDensity( name="phi[6]", dx=-1, dy= 1, group="phi")
AddDensity( name="phi[7]", dx=-1, dy=-1, group="phi")
AddDensity( name="phi[8]", dx= 1, dy=-1, group="phi")

AddDensity( name="g[0]", dx= 0, dy= 0, group="g")
AddDensity( name="g[1]", dx= 1, dy= 0, group="g")
AddDensity( name="g[2]", dx= 0, dy= 1, group="g")
AddDensity( name="g[3]", dx=-1, dy= 0, group="g")
AddDensity( name="g[4]", dx= 0, dy=-1, group="g")
AddDensity( name="g[5]", dx= 1, dy= 1, group="g")
AddDensity( name="g[6]", dx=-1, dy= 1, group="g")
AddDensity( name="g[7]", dx=-1, dy=-1, group="g")
AddDensity( name="g[8]", dx= 1, dy=-1, group="g")

AddDensity( name="f[0]", dx= 0, dy= 0, group="f")
AddDensity( name="f[1]", dx= 1, dy= 0, group="f")
AddDensity( name="f[2]", dx= 0, dy= 1, group="f")
AddDensity( name="f[3]", dx=-1, dy= 0, group="f")
AddDensity( name="f[4]", dx= 0, dy=-1, group="f")
AddDensity( name="f[5]", dx= 1, dy= 1, group="f")
AddDensity( name="f[6]", dx=-1, dy= 1, group="f")
AddDensity( name="f[7]", dx=-1, dy=-1, group="f")
AddDensity( name="f[8]", dx= 1, dy=-1, group="f")

AddDensity( name="h_0[0]", dx= 0, dy= 0, group="h_0")
AddDensity( name="h_0[1]", dx= 1, dy= 0, group="h_0")
AddDensity( name="h_0[2]", dx= 0, dy= 1, group="h_0")
AddDensity( name="h_0[3]", dx=-1, dy= 0, group="h_0")
AddDensity( name="h_0[4]", dx= 0, dy=-1, group="h_0")
AddDensity( name="h_0[5]", dx= 1, dy= 1, group="h_0")
AddDensity( name="h_0[6]", dx=-1, dy= 1, group="h_0")
AddDensity( name="h_0[7]", dx=-1, dy=-1, group="h_0")
AddDensity( name="h_0[8]", dx= 1, dy=-1, group="h_0")

AddDensity( name="h_1[0]", dx= 0, dy= 0, group="h_1")
AddDensity( name="h_1[1]", dx= 1, dy= 0, group="h_1")
AddDensity( name="h_1[2]", dx= 0, dy= 1, group="h_1")
AddDensity( name="h_1[3]", dx=-1, dy= 0, group="h_1")
AddDensity( name="h_1[4]", dx= 0, dy=-1, group="h_1")
AddDensity( name="h_1[5]", dx= 1, dy= 1, group="h_1")
AddDensity( name="h_1[6]", dx=-1, dy= 1, group="h_1")
AddDensity( name="h_1[7]", dx=-1, dy=-1, group="h_1")
AddDensity( name="h_1[8]", dx= 1, dy=-1, group="h_1")

#AddDensity(name="subiter", dx=0, dy=0)

#AddDensity(name="grad_psi[0]", dx=0, dy=0)
#AddDensity(name="grad_psi[1]", dx=0, dy=0)

#AddDensity(name="psi",dx=0, dy=0)


AddQuantity(name="F", vector=T, unit="kgm/s2")
AddQuantity(name="U", vector=T, unit="m/s")
AddQuantity(name="Rho", unit="kg/m3")
AddQuantity(name="n0", unit="An/m3")
AddQuantity(name="n1", unit="An/m3")
#AddQuantity(name="Subiter")
AddQuantity(name="Psi", unit="V")
AddQuantity(name="Phi", unit="V")
AddQuantity(name="GradPsi", vector=T, unit="V/m")
AddQuantity(name="GradPhi", vector=T, unit="V/m")

#AddStage("BaseIteration", "Run", save=Fields$group!="nonono", load=DensityAll$group!="nonono")

#AddStage("CalcPsi", save="psi",load=DensityAll$group == "g")

#AddStage("CalcSubiter", save="subiter",load="subiter")


#AddStage("BaseInit", "Init", save=Fields$group == "g", load=DensityAll$group == "g")

#AddAction("Iteration", c("BaseIteration","CalcPsi", "CalcSubiter"))
#AddAction("Iteration", c("CalcSubiter"))

#AddAction("Init", c("BaseInit"))



AddQuantity(name="rho_e", unit="C/m3")


#AddSetting(name="tau_psi", unit="1", comment='tau_psi') 
#AddSetting(name="tau_phi", unit="1", comment='tau_phi') 

AddSetting(name="n_inf_0", comment='')
AddSetting(name="n_inf_1", comment='')
AddSetting(name="el",  unit="C", comment='')
#AddSetting(name="kb",  unit="J/K", comment='')
#AddSetting(name="T",  unit="K", comment='')
AddSetting(name="el_kbT",  unit="C/J", comment='')
AddSetting(name="epsilon",  unit="C2/J/m", comment='')
AddSetting(name="dt", comment='')

AddSetting(name="psi0",  unit="V", default=1., comment='')
AddSetting(name="phi0",  unit="V", default=1., comment='')


AddSetting(name="ez",  default=1., comment='')
AddSetting(name="Ex",  unit="V/m", default=0, comment='')
AddSetting(name="D",  unit="m2/t", default=1./6., comment='Ion diffusivity')
AddSetting(name="nu",  unit="sPa", comment='viscosity')


AddSetting(name="rho_bc", unit="kg/m3", default=1, comment='fluid density at  boundary', zonal=T)

AddSetting(name="phi_bc", unit="V", default=1, comment='phi at  boundary', zonal=T)
AddSetting(name="psi_bc", unit="V", default=1, comment='psi at  boundary - zeta', zonal=T)
#AddSetting(name="rho0",  unit="kg/m3", comment='density')
AddSetting(name="t_to_s", default="1t/s", unit="t/s", comment ="time scale ratio")

AddGlobal(name="TotalMomentum");

AddNodeType("SSymmetry","BOUNDARY")
AddNodeType("NSymmetry","BOUNDARY")


#AddNodeType("ForceTemperature","ADDITIONALS")
#AddNodeType("ForceConcentration","ADDITIONALS")
#AddNodeType("Seed","ADDITIONALS")


AddNodeType(name="NVelocity", group="BOUNDARY")
AddNodeType(name="SVelocity", group="BOUNDARY")
AddNodeType(name="WVelocity", group="BOUNDARY")
AddNodeType(name="EVelocity", group="BOUNDARY")



AddDensity( name="BC[0]", dx=0, dy=0, group="BC")
AddDensity( name="BC[1]", dx=0, dy=0, group="BC")



