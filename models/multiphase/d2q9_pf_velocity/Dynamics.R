# Density - table of variables of LB Node to stream
#	Pressure Evolution:
AddDensity( name="g[0]", dx= 0, dy= 0, group="g")
AddDensity( name="g[1]", dx= 1, dy= 0, group="g")
AddDensity( name="g[2]", dx= 0, dy= 1, group="g")
AddDensity( name="g[3]", dx=-1, dy= 0, group="g")
AddDensity( name="g[4]", dx= 0, dy=-1, group="g")
AddDensity( name="g[5]", dx= 1, dy= 1, group="g")
AddDensity( name="g[6]", dx=-1, dy= 1, group="g")
AddDensity( name="g[7]", dx=-1, dy=-1, group="g")
AddDensity( name="g[8]", dx= 1, dy=-1, group="g")

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
if (Options$Outflow) {
	AddDensity( name=paste("gold",0:8,sep=""), dx=0, dy=0, group="gold")
	AddDensity( name=paste("hold",0:8,sep=""), dx=0, dy=0, group="hold")
}

#	Fields required for solid contact
AddDensity(name="nw_x", dx=0, dy=0, group="nw")
AddDensity(name="nw_y", dx=0, dy=0, group="nw")

#	Velocity Fields
AddDensity(name="U", dx=0, dy=0, group="Vel")
AddDensity(name="V", dx=0, dy=0, group="Vel")

#	Phase-field stencil for finite differences
AddField('PhaseF',stencil2d=1, group="PF")

#	Additional access required for outflow boundaries
if (Options$Outflow){
	for (d in rows(DensityAll)){
    		AddField( name=d$name,  dx=c(-d$dx-1,-d$dx), dy=c(-d$dy,-d$dy-1) )
	}
	AddField('U',dx=c(-1,0))
	AddField('V',dx=c(0,-1))
}

AddStage("PhaseInit" , "Init" 		, save=Fields$name=="PhaseF")
AddStage("WallInit", "Init_wallNorm"    , save=Fields$group %in% c("nw"))
AddStage("calcWall", "calcWallPhase"    , save=Fields$group %in% c("PF"),
	     				  load=DensityAll$group=="nw") 
if (Options$RT) {
    AddField('PhaseOld')
    AddStage("BaseInit"  , "Init_distributions" , save=Fields$group %in% c("g","h","Vel") )
    AddStage("calcPhase" , "calcPhaseF"		, save=Fields$name %in% c("PhaseF","PhaseOld"), 
						  load=DensityAll$group=="h")
    AddStage("BaseIter"  , "Run" 		, save=Fields$group %in% c("g","h","Vel"), 
						  load=DensityAll$group %in% c("g","h","Vel"))
} else if (Options$Outflow) {
    AddStage("BaseInit"  , "Init_distributions"	, save=Fields$group %in% c("g","h","Vel","gold","hold","PF","nw") )
    AddStage("calcPhase" , "calcPhaseF"		, save=Fields$name %in% c("PhaseF"), 
						  load=DensityAll$group %in% c("g","h","Vel","gold","hold","nw"))
    AddStage("BaseIter"  , "Run" 		, save=Fields$group %in% c("g","h","Vel","gold","hold","nw") , 
						  load=DensityAll$group %in% c("g","h","Vel","gold","hold","nw"))
} else {
    AddStage("BaseInit"  , "Init_distributions"	, save=Fields$group %in% c("g","h","Vel") )
    AddStage("calcPhase" , "calcPhaseF"		, save=Fields$name=="PhaseF", 
						  load=DensityAll$group %in% c("g","h","Vel","nw"))
    AddStage("BaseIter"  , "Run" 		, save=Fields$group %in% c("g","h","Vel","nw") , 
						  load=DensityAll$group %in% c("g","h","Vel","nw"))
}

AddAction("Iteration", c("BaseIter", "calcPhase","calcWall"))
AddAction("Init"     , c("PhaseInit","WallInit", "calcWall","BaseInit"))

# 	Outputs:
AddQuantity(name="Rho",	  unit="kg/m3")
AddQuantity(name="PhaseField",unit="1")
AddQuantity(name="U",	  unit="m/s",vector=T)
AddQuantity(name="P",	  unit="Pa")
AddQuantity(name="Normal", unit="1", vector=T)
#	Initialisation States
AddSetting(name="Period", default="0", comment='Number of cells per cos wave')
AddSetting(name="Perturbation", default="0", comment='Size of wave perturbation, Perturbation Period')
AddSetting(name="MidPoint", default="0", comment='height of RTI centerline')

AddSetting(name="Radius" , default="0", comment='Radius of diffuse interface circle')
AddSetting(name="CenterX", default="0", comment='Circle center x-coord')
AddSetting(name="CenterY", default="0", comment='Circle Center y-coord')
AddSetting(name="BubbleType", default="1", comment='Drop/bubble')

#	Inputs: For phasefield evolution
AddSetting(name="Density_h", comment='High density')
AddSetting(name="Density_l", comment='Low  density')
AddSetting(name="PhaseField_h", default=1, comment='PhaseField in Liquid')
AddSetting(name="PhaseField_l", default=0, comment='PhaseField gas')
AddSetting(name="PhaseField", 	   comment='Initial PhaseField distribution', zonal=T)
AddSetting(name="W", default=4,    comment='Anti-diffusivity coeff')
AddSetting(name="omega_phi", comment='one over relaxation time (phase field)')
AddSetting(name="M", omega_phi='1.0/(3*M+0.5)', default=0.02, comment='Mobility')
AddSetting(name="sigma", 		   comment='surface tension')
AddSetting(name="ContactAngle", radAngle='ContactAngle*3.1415926535897/180', default='90', comment='Contact angle in degrees')
AddSetting(name="radAngle", comment='Conversion to rads for calcs')

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
AddSetting(name="Pressure" , default=0.0, comment='inlet/outlet/init density', zonal=T)
AddSetting(name="GravitationX", default=0.0, comment='applied (rho)*GravitationX')
AddSetting(name="GravitationY", default=0.0, comment='applied (rho)*GravitationY')
AddSetting(name="BuoyancyX", default=0.0, comment='applied (rho-rho_h)*BuoyancyX')
AddSetting(name="BuoyancyY", default=0.0, comment='applied (rho-rho_h)*BuoyancyY')
# Globals - table of global integrals that can be monitored and optimized
AddGlobal(name="PressureLoss", comment='pressure loss', unit="1mPa")
AddGlobal(name="OutletFlux", comment='pressure loss', unit="1m2/s")
AddGlobal(name="InletFlux", comment='pressure loss', unit="1m2/s")
AddGlobal(name="TotalDensity", comment='Mass conservation check', unit="1kg/m3")

AddNodeType(name="SpikeTrack",group="ADDITIONALS")
AddNodeType(name="BubbleTrack",group="ADDITIONALS")
AddGlobal(name="RTIBubble", comment='Bubble Tracker')
AddGlobal(name="RTISpike",  comment='Spike Tracker')

# Boundary things
AddNodeType(name="MovingWall_N", group="BOUNDARY")
AddNodeType(name="MovingWall_S", group="BOUNDARY")
AddNodeType(name="NVelocity", group="BOUNDARY")
AddNodeType(name="SVelocity", group="BOUNDARY")
AddNodeType(name="EVelocity", group="BOUNDARY")
AddNodeType(name="WVelocity", group="BOUNDARY")
AddNodeType(name="NPressure", group="BOUNDARY")
AddNodeType(name="SPressure", group="BOUNDARY")
AddNodeType(name="EPressure", group="BOUNDARY")
AddNodeType(name="WPressure", group="BOUNDARY")

if (Options$Outflow) {
	AddNodeType(name="Convective_E", group="BOUNDARY")
	AddNodeType(name="Convective_N", group="BOUNDARY")
	AddNodeType(name="Neumann_E", group="BOUNDARY")
}
