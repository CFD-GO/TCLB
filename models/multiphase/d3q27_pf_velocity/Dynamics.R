# Density - table of variables of LB Node to stream
#	Velocity-based Evolution d3q27:
if (Options$q27){
	source("d3q27q27.R")
} else {
	source("d3q27q15.R")
}

if (Options$ML){
	for (d in rows(DensityAll)){
		AddQuantity(name=d$name)
	}
}

if (Options$OutFlow){
	AddDensity( name=paste("gold",0:26,sep=""), dx=0,dy=0,dz=0,group="gold")
	AddDensity( name=paste("hold",0:14,sep=""), dx=0,dy=0,dz=0,group="hold")
}

AddDensity(name="pnorm", dx=0, dy=0, dz=0, group="Vel")
AddDensity(name="U", dx=0, dy=0, dz=0, group="Vel")
AddDensity(name="V", dx=0, dy=0, dz=0, group="Vel")
AddDensity(name="W", dx=0, dy=0, dz=0, group="Vel")

AddDensity(name="nw_x", dx=0, dy=0, dz=0, group="nw")
AddDensity(name="nw_y", dx=0, dy=0, dz=0, group="nw")
AddDensity(name="nw_z", dx=0, dy=0, dz=0, group="nw")

save_initial_PF = c("PF")
save_initial    = c("g","h","Vel","PF")
save_iteration  = c("g","h","Vel","nw")
load_iteration  = c("g","h","Vel","nw")
load_phase      = c("g","h","Vel","nw")

if (Options$altContactAngle){
    AddDensity(name="n_k", dx=0, dy=0, dz=0, group="nw")
    AddDensity(name="der_tangent_1_wall", dx=0, dy=0, dz=0)
    AddDensity(name="der_tangent_2_wall", dx=0, dy=0, dz=0)
    # Debugging
    AddDensity(name="perpVal", dx=0, dy=0, dz=0)
    AddField("gradPhiVal_x", stencil3d=2, group="gradPhi")
    AddField("gradPhiVal_y", stencil3d=2, group="gradPhi")
    AddField("gradPhiVal_z", stencil3d=2, group="gradPhi")
    AddField("IsBoundary", stencil3d=1, group="debug_boundary")

    AddDensity(name="TangentWallVector1_x", dx=0, dy=0, dz=0)
    AddDensity(name="TangentWallVector2_x", dx=0, dy=0, dz=0)
    AddDensity(name="TangentWallVector1_y", dx=0, dy=0, dz=0)
    AddDensity(name="TangentWallVector2_y", dx=0, dy=0, dz=0)
    AddDensity(name="TangentWallVector1_z", dx=0, dy=0, dz=0)
    AddDensity(name="TangentWallVector2_z", dx=0, dy=0, dz=0)
    AddField("PhaseF",stencil3d=3, group="PF")

    AddStage("WallInit_CA"  , "Init_wallNorm", save=Fields$group %in% c("nw", "debug_boundary"))
    AddStage("calcWall_CA"  , "calcWallPhase", save=Fields$name %in% c("PhaseF", "der_tangent_1_wall", "der_tangent_2_wall", "TangentWallVector1_x", "TangentWallVector2_x", "TangentWallVector1_y", "TangentWallVector2_y", "TangentWallVector1_z", "TangentWallVector2_z", "perpVal"), load=DensityAll$group %in% c("nw", "gradPhi", "PF"))

    AddStage('calcPhaseGrad', "calcPhaseGrad", load=DensityAll$group %in% c("g","h","Vel","nw", "PF"), save=Fields$group=="gradPhi")
    AddStage('calcPhaseGrad_init', "calcPhaseGrad_init", load=DensityAll$group %in% c("g","h","Vel","nw", "PF"), save=Fields$group=="gradPhi")
} else {
    AddField("PhaseF",stencil3d=1, group="PF")
}
if (Options$OutFlow){
	for (d in rows(DensityAll)) {
		AddField( name=d$name, dx=-d$dx-1, dy=-d$dy, dz=-d$dz )
	}
	AddField(name="U",dx=c(-1,0,0))

    save_initial   = c(save_initial,  "gold","hold")
    save_iteration = c(save_iteration,"gold","hold")
    load_iteration = c(load_iteration,"gold","hold")
    load_phase     = c(load_phase,    "gold","hold")
}
###############################
########THERMOCAPILLARY########
###############################
if (Options$thermo){
    source("thermocapillary.R")

    save_initial_PF = c(save_initial_PF,"Thermal")
    save_iteration  = c(save_iteration, "Thermal")
    load_iteration  = c(load_iteration, "Thermal")
}
######################
########STAGES########
######################
AddStage("WallInit" , "Init_wallNorm", save=Fields$group=="nw")
AddStage("calcWall" , "calcWallPhase", save=Fields$name=="PhaseF", load=DensityAll$group=="nw")
AddStage("PhaseInit", "Init", save=Fields$group %in% save_initial_PF)
AddStage("BaseInit" , "Init_distributions", save=Fields$group %in% save_initial)
AddStage("calcPhase", "calcPhaseF", save=Fields$name=="PhaseF", load=DensityAll$group %in% load_phase)
AddStage("BaseIter" , "Run", save=Fields$group %in% save_iteration, load=DensityAll$group %in% load_iteration )
#######################
########ACTIONS########
#######################
	if (Options$thermo){	
		AddAction("TempToSteadyState", c("CopyDistributions","RK_1", "RK_2", "RK_3", "RK_4","NonLocalTemp"))
		AddAction("Iteration", c("BaseIter", "calcPhase", "calcWall","RK_1", "RK_2", "RK_3", "RK_4","NonLocalTemp"))
		AddAction("IterationConstantTemp", c("BaseIter", "calcPhase", "calcWall","CopyThermal"))
		AddAction("Init"     , c("PhaseInit","WallInit" , "calcWall","BaseInit"))
	} else if (Options$altContactAngle) {
        AddAction("Iteration", c("BaseIter", "calcPhase",    "calcPhaseGrad", "calcWall_CA"))
	    AddAction("Init"     , c("PhaseInit","WallInit_CA" , "calcPhaseGrad_init", "calcWall_CA","BaseInit"))
    } else {
		AddAction("Iteration", c("BaseIter", "calcPhase", "calcWall"))
		AddAction("Init"     , c("PhaseInit","WallInit" , "calcWall","BaseInit"))
	}
#######################
########OUTPUTS########
#######################
	AddQuantity(name="Rho",unit="kg/m3")
	AddQuantity(name="PhaseField",unit="1")
	AddQuantity(name="U",	  unit="m/s",vector=T)
	AddQuantity(name="P",	  unit="Pa")
	AddQuantity(name="Pstar", unit="1")
	AddQuantity(name="Normal", unit=1, vector=T)
if (Options$altContactAngle){
    AddQuantity(name="TangentWallVector1", unit=1, vector=T)
    AddQuantity(name="TangentWallVector2", unit=1, vector=T)
    AddQuantity(name="Tangent1Wall", unit="1")
    AddQuantity(name="Tangent2Wall", unit="1")
    AddQuantity(name="GradPhi", unit=1, vector=T)
    AddQuantity(name="PerpVal", unit="1")
    AddQuantity(name="IsItBoundary", unit="1")
}
###################################
########INPUTS - PHASEFIELD########
###################################
	AddSetting(name="Density_h", comment='High density')
	AddSetting(name="Density_l", comment='Low  density')
	AddSetting(name="PhaseField_h", default=1, comment='PhaseField in Liquid')
	AddSetting(name="PhaseField_l", default=0, comment='PhaseField gas')
	AddSetting(name="PhaseField", 	   comment='Initial PhaseField distribution', zonal=T)
	AddSetting(name="IntWidth", default=4,    comment='Anti-diffusivity coeff')
	AddSetting(name="omega_phi", comment='one over relaxation time (phase field)')
	AddSetting(name="M", omega_phi='1.0/(3*M+0.5)', default=0.02, comment='Mobility')
	AddSetting(name="sigma", comment='surface tension')
  AddSetting(name="Washburn_start", default="0", comment='Start of washburn gas phase')
  AddSetting(name="Washburn_end", default="0", comment='End of washburn gas phase')
	AddSetting(name="radAngle", default='1.570796', comment='Contact angle in radians, can use units -> 90d where d=2pi/360', zonal=T)
	AddSetting(name="minGradient", default='1e-8', comment='if the phase gradient is less than this, set phase normals to zero')
	##SPECIAL INITIALISATIONS
	# RTI
		AddSetting(name="RTI_Characteristic_Length", default=-999, comment='Use for RTI instability')
	    AddSetting(name="pseudo2D", default="0", comment="if 1, assume model is pseduo2D")
    # Single droplet/bubble
		AddSetting(name="Radius", default=0.0, comment='Diffuse Sphere Radius')
		AddSetting(name="CenterX", default=0.0, comment='Diffuse sphere center_x')
		AddSetting(name="CenterY", default=0.0, comment='Diffuse sphere center_y')
		AddSetting(name="CenterZ", default=0.0, comment='Diffuse sphere center_z')
		AddSetting(name="BubbleType",default=1.0, comment='droplet(1.0) or bubble(-1.0)?!')
	# Annular Taylor bubble
		AddSetting(name="DonutTime", default=0.0, comment='Radius of a Torus - initialised to travel along x-axis')
		AddSetting(name="Donut_h",   default=0.0, comment='Half donut thickness, i.e. the radius of the cross-section')
		AddSetting(name="Donut_D",   default=0.0, comment='Dilation factor along the x-axis')
		AddSetting(name="Donut_x0",  default=0.0, comment='Position along x-axis')
	# Poiseuille flow in 2D channel (flow in x direction)
		AddSetting("HEIGHT", default=0,	comment="Height of channel for 2D Poiseuille flow")
		AddSetting("Uavg", default=0,	zonal=T, comment="Average velocity of channel for 2D Poiseuille flow")
		AddSetting("developedFlow", default=0,	comment="set greater than 0 for fully developed flow in the domain (x-direction)")
		AddSetting("developedPipeFlow_X", default=0,	comment="set greater than 0 for fully developed pipe flow in the domain (x-direction)")
        AddSetting("pipeRadius", default=0, comment="radius of pipe for developed pipe flow")
        AddSetting("pipeCentre_Y", default=0, comment="pipe centre Y co-ord for developed pipe flow")
        AddSetting("pipeCentre_Z", default=0, comment="pipe centre Z co-ord for developed pipe flow")
##############################
########INPUTS - FLUID########
##############################
	AddSetting(name="tau_l", comment='relaxation time (low density fluid)')
	AddSetting(name="tau_h", comment='relaxation time (high density fluid)')
    AddSetting(name="tauUpdate", default="1", comment="Interpolation: 1-linear, 2- inverse, 3- dyn viscosity")
	AddSetting(name="Viscosity_l", tau_l='(3*Viscosity_l)', default=0.16666666, comment='kinematic viscosity')
	AddSetting(name="Viscosity_h", tau_h='(3*Viscosity_h)', default=0.16666666, comment='kinematic viscosity')
	AddSetting(name="VelocityX", default=0.0, comment='inlet/outlet/init velocity', zonal=T)
	AddSetting(name="VelocityY", default=0.0, comment='inlet/outlet/init velocity', zonal=T)
	AddSetting(name="VelocityZ", default=0.0, comment='inlet/outlet/init velocity', zonal=T)
	AddSetting(name="Pressure" , default=0.0, comment='inlet/outlet/init density', zonal=T)
	AddSetting(name="GravitationX", default=0.0, comment='applied (rho)*GravitationX')
	AddSetting(name="GravitationY", default=0.0, comment='applied (rho)*GravitationY')
	AddSetting(name="GravitationZ", default=0.0, comment='applied (rho)*GravitationZ')
	AddSetting(name="BuoyancyX", default=0.0, comment='applied (rho_h-rho)*BuoyancyX')
	AddSetting(name="BuoyancyY", default=0.0, comment='applied (rho_h-rho)*BuoyancyY')
	AddSetting(name="BuoyancyZ", default=0.0, comment='applied (rho_h-rho)*BuoyancyZ')
##################################
########TRACKING VARIABLES########
##################################
	AddSetting(name="xyzTrack", default=1,comment='x<-1, y<-2, z<-3')
	AddNodeType("Centerline",group="ADDITIONALS")
	AddNodeType(name="Spiketrack", group="ADDITIONALS")
	AddNodeType(name="Saddletrack", group="ADDITIONALS")
	AddNodeType(name="Bubbletrack", group="ADDITIONALS")
	AddGlobal("InterfacePosition", op="MAX", comment='trackPosition')
	AddGlobal("Vfront",comment='velocity infront of bubble')
	AddGlobal("Vback",comment='velocity behind bubble')
	AddGlobal("RTISpike", op="MAX", comment='SpikeTracker ')
	AddGlobal("RTIBubble",op="MAX", comment='BubbleTracker')
	AddGlobal("RTISaddle",op="MAX", comment='SaddleTracker')
	AddGlobal("XLocation", comment='tracking of x-centroid of the gas regions in domain', unit="m")
	AddGlobal(name="DropFront",	op="MAX",  comment='Highest location of droplet', unit="m")
##########################
########NODE TYPES########
##########################
	AddNodeType("Smoothing",group="ADDITIONALS")
	AddNodeType(name="EPressure", group="BOUNDARY")
	AddNodeType(name="WPressure", group="BOUNDARY")
	AddNodeType(name="NVelocity", group="BOUNDARY")
	AddNodeType(name="Velocity_Y_neg", group="BOUNDARY")
	AddNodeType(name="EVelocity", group="BOUNDARY")
	AddNodeType(name="WVelocity", group="BOUNDARY")
	AddNodeType(name="MovingWall_N", group="BOUNDARY")
	AddNodeType(name="MovingWall_S", group="BOUNDARY")
	AddNodeType(name="Solid", group="BOUNDARY")
	AddNodeType(name="Wall", group="BOUNDARY")
	AddNodeType(name="BGK", group="COLLISION")
	AddNodeType(name="MRT", group="COLLISION")
	if (Options$OutFlow){
		AddNodeType(name="ENeumann", group="BOUNDARY")
		AddNodeType(name="EConvect", group="BOUNDARY")
	}
#######################
########GLOBALS########
#######################
	AddGlobal(name="PressureLoss", comment='pressure loss', unit="1mPa")
	AddGlobal(name="OutletFlux", comment='pressure loss', unit="1m2/s")
	AddGlobal(name="InletFlux", comment='pressure loss', unit="1m2/s")
	AddGlobal(name="TotalDensity", comment='Mass conservation check', unit="1kg/m3")
	AddGlobal(name="KineticEnergy",comment='Measure of kinetic energy', unit="J")
	AddGlobal(name="GasTotalVelocity", comment='use to determine avg velocity of bubbles', unit="m/s")
	AddGlobal(name="GasTotalVelocityX", comment='use to determine avg velocity of bubbles', unit="m/s")
	AddGlobal(name="GasTotalVelocityY", comment='use to determine avg velocity of bubbles', unit="m/s")
	AddGlobal(name="GasTotalVelocityZ", comment='use to determine avg velocity of bubbles', unit="m/s")
	AddGlobal(name="GasTotalPhase",	   comment='use in line with GasTotalVelocity to determine average velocity', unit="1")
	AddGlobal(name="LiqTotalVelocity", 	comment='use to determine avg velocity of droplets', unit="m/s")
	AddGlobal(name="LiqTotalVelocityX", comment='use to determine avg velocity of droplets', unit="m/s")
	AddGlobal(name="LiqTotalVelocityY", comment='use to determine avg velocity of droplets', unit="m/s")
	AddGlobal(name="LiqTotalVelocityZ", comment='use to determine avg velocity of droplets', unit="m/s")
	AddGlobal(name="LiqTotalPhase",	   		comment='use in line with LiqTotalVelocity to determine average velocity', unit="1")
