# Density - table of variables of LB Node to stream
#  name - variable name to stream
#  dx,dy,dz - direction of streaming
#  comment - additional comment
#	Pressure Evolution:
AddDensity( name="f[0]", dx= 0, dy= 0, group="f")
AddDensity( name="f[1]", dx= 1, dy= 0, group="f")
AddDensity( name="f[2]", dx= 0, dy= 1, group="f")
AddDensity( name="f[3]", dx=-1, dy= 0, group="f")
AddDensity( name="f[4]", dx= 0, dy=-1, group="f")
AddDensity( name="f[5]", dx= 1, dy= 1, group="f")
AddDensity( name="f[6]", dx=-1, dy= 1, group="f")
AddDensity( name="f[7]", dx=-1, dy=-1, group="f")
AddDensity( name="f[8]", dx= 1, dy=-1, group="f")
#	Phase Field Evolution:
AddDensity( name="h[0]", dx= 0, dy= 0, group="h")
AddDensity( name="h[1]", dx= 1, dy= 0, group="h")
AddDensity( name="h[2]", dx= 0, dy= 1, group="h")
AddDensity( name="h[3]", dx=-1, dy= 0, group="h")
AddDensity( name="h[4]", dx= 0, dy=-1, group="h")
AddDensity( name="h[5]", dx= 1, dy= 1, group="h")
AddDensity( name="h[6]", dx=-1, dy= 1, group="h")
AddDensity( name="h[7]", dx=-1, dy=-1, group="h")
AddDensity( name="h[8]", dx= 1, dy=-1, group="h")
AddField('PhaseF',stencil2d=2, group="phi")
# Stages - processes to run for initialisation and each iteration
AddStage("PhaseInit"    , "Init"		      , save=Fields$group=="phi")
AddStage("BaseInit"     , "Init_distributions", save=Fields$group=="f" | Fields$group=="h")
AddStage("BaseIter"     , "Run"			      , save=Fields$group=="f" | Fields$group=="h", load=DensityAll$group=="f" | DensityAll$group=="h")
AddStage("calcPhase"	, "calcPhaseF"        , save=Fields$group=="phi"                  , load=DensityAll$group=="h")

AddAction("Iteration", c("BaseIter", "calcPhase" ))
AddAction("Init"     , c("PhaseInit"  ,"BaseInit"))
# 	Outputs:
AddQuantity(name="Rho",unit="kg/m3")
AddQuantity(name="PhaseField",unit="1")
AddQuantity(name="U",unit="m/s",vector=T)
AddQuantity(name="P",unit="Pa")
AddQuantity(name="Mu",unit="1")
AddQuantity(name="Normal",unit="1/m",vector=T)
AddQuantity(name="InterfaceForce", unit="N",vector=T)
#	Inputs: For phasefield evolution
AddSetting(name="Density_h", comment='High density')
AddSetting(name="Density_l", comment='Low  density')
AddSetting(name="PhaseField_h", default=1, comment='PhaseField in Liquid')
AddSetting(name="PhaseField_l", default=0, comment='PhaseField gas')
AddSetting(name="PhaseField", comment='Initial PhaseField distribution', zonal=T)
AddSetting(name="W", default=1, comment='Anti-diffusivity coeff')
AddSetting(name="M", default=1, comment='Mobility')
AddSetting(name="sigma", comment='surface tension')
# 	Inputs: Fluid Properties
AddSetting(name="omega_l", comment='one over relaxation time')
AddSetting(name="omega_h", comment='one over relaxation time')
AddSetting(name="nu_l", omega_l='1.0/(3*nu_l + 0.5)', default=0.16666666, comment='viscosity')
AddSetting(name="nu_h", omega_h='1.0/(3*nu_h + 0.5)', default=0.16666666, comment='viscosity')
AddSetting(name="S0", default=1.0, comment='Relaxation Param')
AddSetting(name="S1", default=1.0, comment='Relaxation Param')
AddSetting(name="S2", default=1.0, comment='Relaxation Param')
AddSetting(name="S3", default=1.0, comment='Relaxation Param')
AddSetting(name="S4", default=1.0, comment='Relaxation Param')
AddSetting(name="S5", default=1.0, comment='Relaxation Param')
AddSetting(name="S6", default=1.0, comment='Relaxation Param')
#	Inputs: Flow Properties
AddSetting(name="VelocityX", default=0.0, comment='inlet/outlet/init velocity', zonal=T)
AddSetting(name="VelocityY", default=0.0, comment='inlet/outlet/init velocity', zonal=T)
AddSetting(name="Pressure" , default=0.0, comment='inlet/outlet/init density', zonal=T)
AddSetting(name="GravitationX", default=0.0)
AddSetting(name="GravitationY", default=0.0)

# Globals - table of global integrals that can be monitored and optimized
AddGlobal(name="PressureLoss", comment='pressure loss', unit="1mPa")
AddGlobal(name="OutletFlux", comment='pressure loss', unit="1m2/s")
AddGlobal(name="InletFlux", comment='pressure loss', unit="1m2/s")

