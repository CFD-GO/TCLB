
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


AddQuantity(name="F", vector=T)
AddQuantity(name="U", vector=T)
AddQuantity(name="Rho")
AddQuantity(name="n0")
AddQuantity(name="n1")
#AddQuantity(name="Subiter")
AddQuantity(name="Psi")
AddQuantity(name="GradPsi", vector=T)


#AddStage("BaseIteration", "Run", save=Fields$group!="nonono", load=DensityAll$group!="nonono")

#AddStage("CalcPsi", save="psi",load=DensityAll$group == "g")

#AddStage("CalcSubiter", save="subiter",load="subiter")


#AddStage("BaseInit", "Init", save=Fields$group == "g", load=DensityAll$group == "g")

#AddAction("Iteration", c("BaseIteration","CalcPsi", "CalcSubiter"))
#AddAction("Iteration", c("CalcSubiter"))

#AddAction("Init", c("BaseInit"))



AddQuantity(name="rho_e",unit="kg/m3")


AddSetting(name="tau_psi", comment='tau_psi')

AddSetting(name="n_inf", comment='')
AddSetting(name="el", comment='')
AddSetting(name="kb", comment='')
AddSetting(name="T", comment='')
AddSetting(name="epsilon", comment='')
AddSetting(name="dt", comment='')
AddSetting(name="psi0", default=1., comment='')
AddSetting(name="ez", default=1., comment='')
AddSetting(name="Ex", default=0, comment='')
AddSetting(name="omega", default=1., comment='')



AddSetting(name="psi_bc", default=1, comment='psi at  boundary - zeta', zonal=T)


#AddGlobal(name="OutFlux");

#AddNodeType("Heater","ADDITIONALS")
#AddNodeType("ForceTemperature","ADDITIONALS")
#AddNodeType("ForceConcentration","ADDITIONALS")
#AddNodeType("Seed","ADDITIONALS")

