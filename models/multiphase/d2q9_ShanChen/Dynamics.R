
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

AddDensity( name="rho", dx=0, dy=0, group="density") # how to run it using AddField("rho", dx=c(0,0), dy=c(0,0), group="density")

AddField("psi", stencil2d=1, group="pp") # Pseudopotential field
AddField("neighbour_type", stencil2d=1, group="neighbour_type_group")

# Stages and Actions
# Initialization list
AddStage("BaseInit"     , "Init", save=Fields$group %in% c("f", "density", "neighbour_type_group"), load=DensityAll$group%in% c("f","density"))

# Iteration list
AddStage("BaseIteration", "Run"     ,  save=Fields$group %in% c("f", "density", "neighbour_type_group") , load=DensityAll$group %in% c("f","density","neighbour_type_group"))
AddStage("PsiIteration" , "calcPsi" ,  save=Fields$name=="psi", load=DensityAll$group %in% c("f", "density"))

AddAction("Init"     , c("BaseInit",      "PsiIteration"))
AddAction("Iteration", c("BaseIteration", "PsiIteration"))

# Output Values
AddQuantity( name="U",    unit="m/s", vector=TRUE )
AddQuantity( name="Rho",  unit="kg/m3" )
AddQuantity( name="Psi",  unit="1" )

# Model Specific Parameters
AddSetting( name="omega", comment='inverse of relaxation time')
AddSetting( name="viscosity", omega='1.0/(3*viscosity+0.5)', default=0.16666666, comment='kinematic viscosity')
AddSetting( name="VelocityX",default=0, comment='inlet/outlet/init velocity', zonal=TRUE)
AddSetting( name="VelocityY",default=0, comment='inlet/outlet/init velocity', zonal=TRUE)
AddSetting( name="GravitationX",default=0, comment='body/external acceleration', zonal=TRUE)
AddSetting( name="GravitationY",default=0, comment='body/external acceleration', zonal=TRUE)
AddSetting( name="Density",default=1, comment='Density',zonal=TRUE)
AddNodeType(name="MovingWall", group="BOUNDARY")

AddSetting( name="G_ff",default=0, comment='fluid-fluid interaction strength')
AddSetting( name="G_sf",default=0, comment='solid-fluid interaction strength')
