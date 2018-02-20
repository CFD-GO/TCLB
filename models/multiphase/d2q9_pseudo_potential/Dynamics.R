
# Fluid Density Populations
AddDensity( name="f[0]", dx= 0, dy= 0, group="f")
AddDensity( name="f[1]", dx= 1, dy= 0, group="f")
AddDensity( name="f[2]", dx= 0, dy= 1, group="f")
AddDensity( name="f[3]", dx=-1, dy= 0, group="f")
AddDensity( name="f[4]", dx= 0, dy=-1, group="f")
AddDensity( name="f[5]", dx= 1, dy= 1, group="f")
AddDensity( name="f[6]", dx=-1, dy= 1, group="f")
AddDensity( name="f[7]", dx=-1, dy=-1, group="f")
AddDensity( name="f[8]", dx= 1, dy=-1, group="f")

# Pseudopotential field
AddField("psi", stencil2d=1, group="pp")
AddField("neighbour_type", stencil2d=1, group="neighbour_type_group")

# Stages and Actions

# Initialization list
AddStage("BaseInit"     , "Init", save=Fields$group %in% c("f", "neighbour_type_group"), load=DensityAll$group=="f")

# Iteration list
AddStage("BaseIteration", "Run"     ,  save=Fields$group %in% c("f", "neighbour_type_group") , load=DensityAll$group %in% c("f","neighbour_type_group"))
AddStage("PsiIteration" , "calcPsi" ,  save=Fields$name=="psi", load=DensityAll$group %in% c("f"))

AddAction("Init"     , c("BaseInit",      "PsiIteration"))
AddAction("Iteration", c("BaseIteration", "PsiIteration"))

# Output Values
AddQuantity( name="U",    unit="m/s", vector=TRUE )
AddQuantity( name="Ueq",  unit="m/s", vector=TRUE )
AddQuantity( name="Rho",  unit="kg/m3" )
AddQuantity( name="Psi",  unit="1" )
AddQuantity( name="Neighbour_type",  unit="1" )
AddQuantity( name="F_ff", unit="N", vector=TRUE)
AddQuantity( name="F_sf", unit="N", vector=TRUE)


# Model Specific Parameters
AddSetting( name="omega", comment='inverse of relaxation time')
AddSetting( name="nu", omega='1.0/(3*nu+0.5)', default=0.16666666, comment='viscosity')
AddSetting( name="VelocityX",default=0, comment='inlet/outlet/init velocity', zonal=TRUE)
AddSetting( name="VelocityY",default=0, comment='inlet/outlet/init velocity', zonal=TRUE)
AddSetting( name="GravitationX",default=0, comment='body/external acceleration', zonal=TRUE)
AddSetting( name="GravitationY",default=0, comment='body/external acceleration', zonal=TRUE)
AddSetting( name="Density",default=1, comment='Density',zonal=TRUE)

AddSetting( name="G_ff",default=0, comment='fluid-fluid interaction strength')
AddSetting( name="G_sf",default=0, comment='fluid-solid interaction strength')