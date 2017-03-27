
AddDensity( name="g[0]", dx= 0, dy= 0, group="g")
AddDensity( name="g[1]", dx= 1, dy= 0, group="g")
AddDensity( name="g[2]", dx= 0, dy= 1, group="g")
AddDensity( name="g[3]", dx=-1, dy= 0, group="g")
AddDensity( name="g[4]", dx= 0, dy=-1, group="g")
AddDensity( name="g[5]", dx= 1, dy= 1, group="g")
AddDensity( name="g[6]", dx=-1, dy= 1, group="g")
AddDensity( name="g[7]", dx=-1, dy=-1, group="g")
AddDensity( name="g[8]", dx= 1, dy=-1, group="g")

AddDensity(name="subiter")



AddField(name="psi",stencil2d=1)


AddQuantity(name="Psi")
AddQuantity(name="Subiter")


AddStage("BaseIteration", "Run", save=Fields$group == "g", load=DensityAll$group == "g")

AddStage("CalcPsi", save="psi",load=DensityAll$group == "g")

AddStage("CalcSubiter", save="subiter",load="subiter")


#AddStage("BaseInit", "Init", save=Fields$group == "g", load=DensityAll$group == "g")

AddAction("Iteration", c("BaseIteration","CalcPsi", "CalcSubiter"))
#AddAction("Iteration", c("CalcSubiter"))

#AddAction("Init", c("BaseInit"))



AddQuantity(name="rho_e",unit="kg/m3")


AddSetting(name="tau_psi", comment='tau_psi')

AddSetting(name="n_inf", comment='')
AddSetting(name="z", comment='')
AddSetting(name="el", comment='')
AddSetting(name="kb", comment='')
AddSetting(name="T", comment='')
AddSetting(name="epsilon", comment='')
AddSetting(name="dt", comment='')

AddSetting(name="psi_bc", default=1, comment='psi at  boundary - zeta', zonal=T)
AddSetting(name="psi0", default=1, comment='initial psi - zeta', zonal=T)


#AddGlobal(name="OutFlux");

#AddNodeType("Heater","ADDITIONALS")
#AddNodeType("ForceTemperature","ADDITIONALS")
#AddNodeType("ForceConcentration","ADDITIONALS")
#AddNodeType("Seed","ADDITIONALS")

