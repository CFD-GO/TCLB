# Setting permissive access policy.
#  * This skips checks of fields being overwritten or read prematurely.
#  * Otherwise the model compilation was failing.
#  * This should be removed if the issue is fixed
SetOptions(permissive.access=TRUE)  ### WARNING


# 	Density - table of variables of LB Node to stream
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
AddDensity(name="nw_x", dx=0, dy=0, group="nw", comment='phase field normal at the wall in x direction, pointing into fluid')
AddDensity(name="nw_y", dx=0, dy=0, group="nw")

#	Velocity Fields
AddDensity(name="U", dx=0, dy=0, group="Vel")
AddDensity(name="V", dx=0, dy=0, group="Vel")

#	Phase-field stencil for finite differences
AddField('PhaseF',stencil2d=1, group="PF")

#	Additional access required for outflow boundaries
if (Options$Outflow){
	for (d in rows(DensityAll)){
    		AddField( name=d$name,  dx=-d$dx-1, dy=-d$dy )
    		AddField( name=d$name,  dx=-d$dx, dy=-d$dy-1 )
	}
	AddField('U',dx=c(-1,0))
	AddField('V',dx=c(0,-1))
}

#	Stages - processes to run for initialisation and each iteration
if (Options$RT) {
 	AddField('PhaseOld', group="PF")

	# initialisation
	AddStage("PhaseInit" , "Init_phase" 		, save=Fields$group %in% c("PF"))
	AddStage("WallInit"  , "Init_wallNorm"    	, save=Fields$group %in% c("nw"))
	AddStage("BaseInit"  , "Init_distributions" , save=Fields$group %in% c("g","h","Vel"))
	
	# iteration
	AddStage("BaseIter"  , "calcHydroIter" 		, save=Fields$group %in% c("g","h","Vel")		, load=DensityAll$group %in% c("g","h","Vel"))
    AddStage("PhaseIter" , "calcPhaseFIter"		, save=Fields$name  %in% c("PhaseF","PhaseOld") , load=DensityAll$group=="h")
  	AddStage("WallIter"  , "calcWallPhaseIter"  , save=Fields$group %in% c("PF")				, load=DensityAll$group=="nw") 
} else if (Options$Outflow) {

	# initialisation
	AddStage("PhaseInit" , "Init_phase"			, save=Fields$group %in% c("PF"))
	AddStage("WallInit"  , "Init_wallNorm"		, save=Fields$group %in% c("nw"))
	AddStage("BaseInit"  , "Init_distributions"	, save=Fields$group %in% c("g","h","Vel","gold","hold")) 
	# iteration
	AddStage("BaseIter"  , "calcHydroIter"      , save=Fields$group %in% c("g","h","Vel","nw","gold","hold"), 
												  load=DensityAll$group %in% c("g","h","Vel","nw","gold","hold")) 
	AddStage("PhaseIter" , "calcPhaseFIter"		, save=Fields$group %in% c("PF"), load=DensityAll$group %in% c("g","h","Vel","nw","gold","hold"))
	AddStage("WallIter"  , "calcWallPhaseIter"	, save=Fields$group %in% c("PF"), load=DensityAll$group=="nw")	
} else {
	
	# initialisation
	AddStage("PhaseInit" , "Init_phase"			, save=Fields$group %in% c("PF"))
	AddStage("WallInit"  , "Init_wallNorm"		, save=Fields$group %in% c("nw"))
	AddStage("BaseInit"  , "Init_distributions" , save=Fields$group %in% c("g","h","Vel"))

	# iteration
	AddStage("BaseIter"  , "calcHydroIter"      , save=Fields$group %in% c("g","h","Vel","nw") , load=DensityAll$group %in% c("PF","g","h","Vel","nw"))  # TODO: is nw needed here?
	AddStage("PhaseIter" , "calcPhaseFIter"		, save=Fields$group %in% c("PF")			   , load=DensityAll$group %in% c("g","h","Vel","nw"))
	AddStage("WallIter"  , "calcWallPhaseIter"	, save=Fields$group %in% c("PF")			   , load=DensityAll$group %in% c("nw","PF"))	# Purposefully read/write of PF for boundary. complex geom may force RACE condition.
}

AddAction("Iteration", c("BaseIter", "PhaseIter","WallIter"))
AddAction("Init"     , c("PhaseInit","WallInit", "WallIter","BaseInit"))

# 	Outputs:
AddQuantity(name="Rho",	unit="kg/m3")
AddQuantity(name="PhaseField", unit="1")
AddQuantity(name="U", unit="m/s",vector=T)
AddQuantity(name="NormalizedPressure", unit="1")
AddQuantity(name="Pressure", unit="Pa")
AddQuantity(name="Normal", unit="1", vector=T)

#	Initialisation States
AddSetting(name="Period", default="0", comment='Number of cells per cos wave')
AddSetting(name="Perturbation", default="0", comment='Size of wave perturbation, Perturbation Period')
AddSetting(name="MidPoint", default="0", comment='height of RTI centerline')

AddSetting(name="Wave", default="0", comment='Used for gravity and capillary wave benchmarks')

AddSetting(name="Radius" , default="0", comment='Radius of diffuse interface circle')
AddSetting(name="CenterX", default="0", comment='Circle center x-coord')
AddSetting(name="CenterY", default="0", comment='Circle Center y-coord')
AddSetting(name="BubbleType", default="1", comment='Drop/bubble')

#	Inputs: For phasefield evolution
AddSetting(name="Density_h", comment='High density fluid')
AddSetting(name="Density_l", comment='Low  density fluid')
AddSetting(name="PhaseField_h", default=1, comment='PhaseField in high density fluid')
AddSetting(name="PhaseField_l", default=0, comment='PhaseField in low density fluid')
AddSetting(name="PhaseField_init", 	   comment='Initial/Inflow PhaseField distribution', zonal=T)
AddSetting(name="W", default=4,    comment='Anti-diffusivity coeff (phase interfacial thickness) ')
AddSetting(name="omega_phi", comment='one over relaxation time (phase field)')
AddSetting(name="M", omega_phi='1.0/(3*M+0.5)', default=0.02, comment='Mobility')
AddSetting(name="sigma", 		   comment='surface tension')
AddSetting(name="radAngle",default='1.570796', comment='Contact angle in radians, can use units -> 90d where d=2pi/360', zonal=T)

# 	Inputs: Fluid Properties
AddSetting(name="tau_l", comment='relaxation time (low density fluid)')
AddSetting(name="tau_h", comment='relaxation time (high density fluid)')
AddSetting(name="Viscosity_l", tau_l='(3*Viscosity_l)', default=0.16666666, comment='kinematic viscosity')
AddSetting(name="Viscosity_h", tau_h='(3*Viscosity_h)', default=0.16666666, comment='kinematic viscosity')

AddSetting(name="omega_bulk", comment='inverse of bulk relaxation time', default=1.0)
AddSetting(name="bulk_visc", omega_bulk='1.0/(3*bulk_visc+0.5)',  comment='bulk viscosity')

#	Inputs: Flow Properties
AddSetting(name="VelocityX", default=0.0, comment='inlet/outlet/init velocity', zonal=T)
AddSetting(name="VelocityY", default=0.0, comment='inlet/outlet/init velocity', zonal=T)
AddSetting(name="Pressure" , default=0.0, comment='inlet/outlet/init density', zonal=T)
AddSetting(name="GravitationX", default=0.0, comment='applied (rho)*GravitationX', zonal=T)
AddSetting(name="GravitationY", default=0.0, comment='applied (rho)*GravitationY', zonal=T)
AddSetting(name="BuoyancyX", default=0.0, comment='applied (rho-rho_h)*BuoyancyX')
AddSetting(name="BuoyancyY", default=0.0, comment='applied (rho-rho_h)*BuoyancyY')
AddSetting(name="fixedIterator", default=2.0, comment='fixed iterator for velocity calculation')
#	Globals - table of global integrals that can be monitored and optimized
AddGlobal(name="PressureLoss", comment='pressure loss', unit="1mPa")
AddGlobal(name="OutletFlux", comment='pressure loss', unit="1m2/s")
AddGlobal(name="InletFlux", comment='pressure loss', unit="1m2/s")
AddGlobal(name="TotalDensity", comment='Mass conservation check', unit="1kg/m3")
AddNodeType(name="SpikeTrack", group="ADDITIONALS")
AddNodeType(name="BubbleTrack", group="ADDITIONALS")
AddNodeType(name="WaveTrack", group="ADDITIONALS")
AddGlobal(name="RTIBubble", comment='Bubble Tracker', op="MAX")
AddGlobal(name="RTISpike",  comment='Spike Tracker', op="MAX")
AddGlobal(name="WaveLocation", comment='Wave', op="MAX")
AddGlobal(name="NMovingWallForce", comment='force exerted on the N Moving Wall')
AddGlobal(name="NMovingWallPower", comment='implented: Vx* incoming momentum (precollision)')

AddGlobal(name="BubbleVelocityX", comment='Bubble velocity in the x direction')
AddGlobal(name="BubbleVelocityY", comment='Bubble velocity in the y direction')
AddGlobal(name="BubbleVelocityZ", comment='Bubble velocity in the z direction')
AddGlobal(name="BubbleLocationY", comment='Bubble Location in the y direction')
AddGlobal(name="SumPhiGas", comment='Summation of (1-phi) in all gas cells')

if (Options$debug){
	AddGlobal(name="MomentumX", comment='Total momentum in the domain', unit="")
	AddGlobal(name="MomentumY", comment='Total momentum in the domain', unit="")
	AddGlobal(name="MomentumX_afterCol", comment='Total momentum in the domain', unit="")
	AddGlobal(name="MomentumY_afterCol", comment='Total momentum in the domain', unit="")

	AddGlobal(name="F_pressureX", comment='Pressure force X', unit="")
	AddGlobal(name="F_pressureY", comment='Pressure force Y', unit="")
	AddGlobal(name="F_bodyX", comment='Body force X', unit="")
	AddGlobal(name="F_bodyY", comment='Body force Y', unit="")
	AddGlobal(name="F_surf_tensionX", comment='Surface tension force X', unit="")
	AddGlobal(name="F_surf_tensionY", comment='Surface tension force Y', unit="")
	AddGlobal(name="F_muX", comment='Viscous tension force X', unit="")
	AddGlobal(name="F_muY", comment='Viscous tension force Y', unit="")
	AddGlobal(name="F_total_hydroX", comment='Total hydrodynamic force X', unit="")
	AddGlobal(name="F_total_hydroY", comment='Total hydrodynamic force Y', unit="")

	AddGlobal(name="F_phiX", comment='Forcing term for interface tracking X', unit="")
	AddGlobal(name="F_phiY", comment='Forcing term for interface tracking Y', unit="")
}
#	Node things
if (Options$CM){
	AddNodeType(name="CM", group="COLLISION")  # Central Moments collision
}
AddNodeType(name="Smoothing", group="ADDITIONALS")  #  To smooth phase field interface during initialization.

#	Boundary things
AddNodeType(name="MovingWall_N", group="BOUNDARY")
AddNodeType(name="MovingWall_S", group="BOUNDARY")
AddNodeType(name="NVelocity", group="BOUNDARY")
AddNodeType(name="WVelocity", group="BOUNDARY")

AddNodeType(name="Body", group="BODY")  # To measure force exerted on the body.

AddGlobal(name="FDrag", comment='Force exerted on body in X-direction', unit="N")
AddGlobal(name="FLift", comment='Force exerted on body in Y-direction', unit="N")
AddGlobal(name="FTotal", comment='Force exerted on body in X+Y -direction', unit="N")

if (Options$Outflow) {
	AddNodeType(name="Convective_E", group="BOUNDARY")
	AddNodeType(name="Convective_N", group="BOUNDARY")
	AddNodeType(name="Neumann_E", group="BOUNDARY")
}
AddNodeType(name="Solid", group="BOUNDARY")
AddNodeType(name="Wall", group="BOUNDARY")
AddNodeType(name="MRT", group="COLLISION")
