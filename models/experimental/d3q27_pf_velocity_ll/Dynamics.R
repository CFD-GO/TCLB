# Density - table of variables of LB Node to stream
#	Velocity-based Evolution d3q27:
AddDensity( name="g0", dx= 0, dy= 0, dz= 0, group="g")
AddDensity( name="g1", dx= 1, dy= 0, dz= 0, group="g")
AddDensity( name="g2", dx=-1, dy= 0, dz= 0, group="g")
AddDensity( name="g3", dx= 0, dy= 1, dz= 0, group="g")
AddDensity( name="g4", dx= 0, dy=-1, dz= 0, group="g")
AddDensity( name="g5", dx= 0, dy= 0, dz= 1, group="g")
AddDensity( name="g6", dx= 0, dy= 0, dz=-1, group="g")
AddDensity( name="g7", dx= 1, dy= 1, dz= 1, group="g")
AddDensity( name="g8", dx=-1, dy= 1, dz= 1, group="g")
AddDensity( name="g9", dx= 1, dy=-1, dz= 1, group="g")
AddDensity( name="g10",dx=-1, dy=-1, dz= 1, group="g")
AddDensity( name="g11",dx= 1, dy= 1, dz=-1, group="g")
AddDensity( name="g12",dx=-1, dy= 1, dz=-1, group="g")
AddDensity( name="g13",dx= 1, dy=-1, dz=-1, group="g")
AddDensity( name="g14",dx=-1, dy=-1, dz=-1, group="g")
AddDensity( name="g15",dx= 1, dy= 1, dz= 0, group="g")
AddDensity( name="g16",dx=-1, dy= 1, dz= 0, group="g")
AddDensity( name="g17",dx= 1, dy=-1, dz= 0, group="g")
AddDensity( name="g18",dx=-1, dy=-1, dz= 0, group="g")
AddDensity( name="g19",dx= 1, dy= 0, dz= 1, group="g")
AddDensity( name="g20",dx=-1, dy= 0, dz= 1, group="g")
AddDensity( name="g21",dx= 1, dy= 0, dz=-1, group="g")
AddDensity( name="g22",dx=-1, dy= 0, dz=-1, group="g")
AddDensity( name="g23",dx= 0, dy= 1, dz= 1, group="g")
AddDensity( name="g24",dx= 0, dy=-1, dz= 1, group="g")
AddDensity( name="g25",dx= 0, dy= 1, dz=-1, group="g")
AddDensity( name="g26",dx= 0, dy=-1, dz=-1, group="g")

#	Phase Field Evolution d3q15:
AddDensity( name="h0", dx= 0, dy= 0, dz= 0, group="h")
AddDensity( name="h1", dx= 1, dy= 0, dz= 0, group="h")
AddDensity( name="h2", dx=-1, dy= 0, dz= 0, group="h")
AddDensity( name="h3", dx= 0, dy= 1, dz= 0, group="h")
AddDensity( name="h4", dx= 0, dy=-1, dz= 0, group="h")
AddDensity( name="h5", dx= 0, dy= 0, dz= 1, group="h")
AddDensity( name="h6", dx= 0, dy= 0, dz=-1, group="h")
AddDensity( name="h7", dx= 1, dy= 1, dz= 1, group="h")
AddDensity( name="h8", dx=-1, dy= 1, dz= 1, group="h")
AddDensity( name="h9", dx= 1, dy=-1, dz= 1, group="h")
AddDensity( name="h10",dx=-1, dy=-1, dz= 1, group="h")
AddDensity( name="h11",dx= 1, dy= 1, dz=-1, group="h")
AddDensity( name="h12",dx=-1, dy= 1, dz=-1, group="h")
AddDensity( name="h13",dx= 1, dy=-1, dz=-1, group="h")
AddDensity( name="h14",dx=-1, dy=-1, dz=-1, group="h")

AddDensity(name="U", dx=0, dy=0, dz=0, group="Vel")
AddDensity(name="V", dx=0, dy=0, dz=0, group="Vel")
AddDensity(name="W", dx=0, dy=0, dz=0, group="Vel")

AddField('PhaseF',stencil3d=1, group="OrderParameter")

# Stages - processes to run for initialisation and each iteration
AddStage("PhaseInit"    , "Init", save="PhaseF")
AddStage("BaseInit"     , "Init_distributions", save=Fields$group=="g" | Fields$group=="h" | Fields$group=="Vel")
AddStage("calcPhase"	, "calcPhaseF",	save='PhaseF'                             , 
	load=DensityAll$group=="h")
AddStage("BaseIter"     , "Run" , save=Fields$group=="g" | Fields$group=="h" | Fields$group=="Vel", 
	load=DensityAll$group=="g" | DensityAll$group=="h" | DensityAll$group=="Vel")

if (Options$SC) {
AddStage("WallPhase", "calcWallPhase", save="PhaseF")
AddAction("Iteration", c("BaseIter", "calcPhase", "WallPhase"))
AddAction("Init"     , c("PhaseInit", "WallPhase","BaseInit", "calcPhase"))
} else {
AddAction("Iteration", c("BaseIter", "calcPhase"))
AddAction("Init"     , c("PhaseInit","BaseInit", "calcPhase"))
}

# 	Outputs:
AddQuantity(name="PhaseField",unit="1")
AddQuantity(name="U",	  unit="m/s",vector=T)
AddQuantity(name="P",	  unit="Pa")

#	Inputs: For phasefield evolution
AddSetting(name="Density_h", comment='High density')
AddSetting(name="Density_l", comment='Low  density')
AddSetting(name="PhaseField_h", default=1, comment='PhaseField in Liquid')
AddSetting(name="PhaseField_l", default=0, comment='PhaseField gas')
AddSetting(name="PhaseField", 	   comment='Initial PhaseField distribution', zonal=T)
AddSetting(name="IntWidth", default=4,    comment='Anti-diffusivity coeff')
AddSetting(name="omega_phi", comment='one over relaxation time (phase field)')
AddSetting(name="M", omega_phi='1.0/(3*M+0.5)', default=0.02, comment='Mobility')
AddSetting(name="sigma", 		   comment='surface tension')

if (Options$SC) {
AddSetting(name="ContactAngle", default="90", comment='Contact angle of the phases')
}

#Domain initialisation (pre-defined set-ups)
AddSetting(name="RTI_Characteristic_Length", default=-999, comment='Use for RTI instability')

AddSetting(name="Radius", default="0.0", comment='Diffuse Sphere Radius')
AddSetting(name="CenterX", default="0.0", comment='Diffuse sphere center_x')
AddSetting(name="CenterY", default="0.0", comment='Diffuse sphere center_y')
AddSetting(name="CenterZ", default="0.0", comment='Diffuse sphere center_z')
AddSetting(name="BubbleType",default="1.0", comment='droplet or bubble?!')

# 	Inputs: Fluid Properties
AddSetting(name="tau_l", comment='relaxation time (low density fluid)')
AddSetting(name="tau_h", comment='relaxation time (high density fluid)')
AddSetting(name="Viscosity_l", tau_l='(3*Viscosity_l)', default=0.16666666, comment='kinematic viscosity')
AddSetting(name="Viscosity_h", tau_h='(3*Viscosity_h)', default=0.16666666, comment='kinematic viscosity')
#AddSetting(name="S0", default=1.0, comment='Relaxation Param')
#AddSetting(name="S1", default=1.0, comment='Relaxation Param')
#AddSetting(name="S2", default=1.0, comment='Relaxation Param')
#AddSetting(name="S3", default=1.0, comment='Relaxation Param')
#AddSetting(name="S4", default=1.0, comment='Relaxation Param')
#AddSetting(name="S5", default=1.0, comment='Relaxation Param')
#AddSetting(name="S6", default=1.0, comment='Relaxation Param')
#	Inputs: Flow Properties
AddSetting(name="VelocityX", default=0.0, comment='inlet/outlet/init velocity', zonal=T)
AddSetting(name="VelocityY", default=0.0, comment='inlet/outlet/init velocity', zonal=T)
AddSetting(name="VelocityZ", default=0.0, comment='inlet/outlet/init velocity', zonal=T)
AddSetting(name="Pressure" , default=0.0, comment='inlet/outlet/init density', zonal=T)
AddSetting(name="GravitationX", default=0.0, comment='applied (rho)*GravitationX')
AddSetting(name="GravitationY", default=0.0, comment='applied (rho)*GravitationY')
AddSetting(name="GravitationZ", default=0.0, comment='applied (rho)*GravitationZ')
AddSetting(name="BuoyancyX", default=0.0, comment='applied (rho-rho_h)*BuoyancyX')
AddSetting(name="BuoyancyY", default=0.0, comment='applied (rho-rho_h)*BuoyancyY')
AddSetting(name="BuoyancyZ", default=0.0, comment='applied (rho-rho_h)*BuoyancyZ')

# Velocity Tracking on Centerline:
#  For TaylorBubble tracking
AddNodeType("Centerline",group="ADDITIONALS")
#  For RTI interface tracking
AddNodeType("Spiketrack",group="ADDITIONALS")
AddNodeType("Saddletrack",group="ADDITIONALS")
AddNodeType("Bubbletrack",group="ADDITIONALS")

AddNodeType(name="MovingWall_N", group="BOUNDARY")
AddNodeType(name="MovingWall_S", group="BOUNDARY")
AddNodeType(name="SymmetricXY_W",group="ADDITIONALS")
AddNodeType(name="SymmetricXY_E",group="ADDITIONALS")

AddGlobal("InterfacePosition",comment='trackPosition')
AddGlobal("Vfront",comment='velocity infront of bubble')
AddGlobal("Vback",comment='velocity behind bubble')
AddGlobal("RTISpike", comment='SpikeTracker ')
AddGlobal("RTIBubble",comment='BubbleTracker')
AddGlobal("RTISaddle",comment='SaddleTracker')

# Globals - table of global integrals that can be monitored and optimized
AddGlobal(name="PressureLoss", comment='pressure loss', unit="1mPa")
AddGlobal(name="OutletFlux", comment='pressure loss', unit="1m2/s")
AddGlobal(name="InletFlux", comment='pressure loss', unit="1m2/s")
AddGlobal(name="TotalDensity", comment='Mass conservation check', unit="1kg/m3")


