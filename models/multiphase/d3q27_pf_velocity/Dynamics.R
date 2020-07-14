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

if (Options$OutFlow){
	AddDensity( name=paste("gold",0:26,sep=""), dx=0,dy=0,dz=0,group="gold")
	AddDensity( name=paste("hold",0:14,sep=""), dx=0,dy=0,dz=0,group="hold")
}

AddDensity(name="U", dx=0, dy=0, dz=0, group="Vel")
AddDensity(name="V", dx=0, dy=0, dz=0, group="Vel")
AddDensity(name="W", dx=0, dy=0, dz=0, group="Vel")

AddDensity(name="nw_x", dx=0, dy=0, dz=0, group="nw")
AddDensity(name="nw_y", dx=0, dy=0, dz=0, group="nw")
AddDensity(name="nw_z", dx=0, dy=0, dz=0, group="nw")

AddField("PhaseF",stencil3d=1, group="PF")

if (Options$OutFlow){
	for (d in rows(DensityAll)) {
		AddField( name=d$name, dx=-d$dx-1, dy=-d$dy, dz=-d$dz )
	}
	AddField(name="U",dx=c(-1,0,0))
}

if (Options$thermo){
	AddField( name="g0" , group="g")
	AddField( name="g1" , group="g")
	AddField( name="g2" , group="g")
	AddField( name="g3" , group="g")
	AddField( name="g4" , group="g")
	AddField( name="g5" , group="g")
	AddField( name="g6" , group="g")
	AddField( name="g7" , group="g")
	AddField( name="g8" , group="g")
	AddField( name="g9" , group="g")
	AddField( name="g10", group="g")
	AddField( name="g11", group="g")
	AddField( name="g12", group="g")
	AddField( name="g13", group="g")
	AddField( name="g14", group="g")
	AddField( name="g15", group="g")
	AddField( name="g16", group="g")
	AddField( name="g17", group="g")
	AddField( name="g18", group="g")
	AddField( name="g19", group="g")
	AddField( name="g20", group="g")
	AddField( name="g21", group="g")
	AddField( name="g22", group="g")
	AddField( name="g23", group="g")
	AddField( name="g24", group="g")
	AddField( name="g25", group="g")
	AddField( name="g26", group="g")
	AddField( name="h0", group="h")
	AddField( name="h1", group="h")
	AddField( name="h2", group="h")
	AddField( name="h3", group="h")
	AddField( name="h4", group="h")
	AddField( name="h5", group="h")
	AddField( name="h6", group="h")
	AddField( name="h7", group="h")
	AddField( name="h8", group="h")
	AddField( name="h9", group="h")
	AddField( name="h10",group="h")
	AddField( name="h11",group="h")
	AddField( name="h12",group="h")
	AddField( name="h13",group="h")
	AddField( name="h14",group="h")
# Temperature Related alterations
	AddDensity("Temp", dx=0, dy=0, dz=0, group="Thermal")
	AddDensity("Cond", dx=0, dy=0, dz=0, group="Thermal")
	AddDensity("SurfaceTension", dx=0, dy=0, dz=0, group="Thermal")
	AddField("Temp",stencil3d=1, group="Thermal")
	AddField("Cond",stencil3d=1, group="Thermal")
	AddField("SurfaceTension",stencil3d=1, group="Thermal")

	#AddDensity("gradPhix", dx=0, dy=0, dz=0, group="Thermal")
	#AddDensity("gradPhiy", dx=0, dy=0, dz=0, group="Thermal")
	#AddDensity("gradPhiz", dx=0, dy=0, dz=0, group="Thermal")
	#AddField("gradPhix",stencil3d=1, group="Thermal")
	#AddField("gradPhiy",stencil3d=1, group="Thermal")
	#AddField("gradPhiz",stencil3d=1, group="Thermal")

	AddQuantity(name="T",unit="K")
	AddSetting("sigma_T",			comment="Derivative describing how surface tension changes with temp")
	AddSetting("T_ref",				comment="Reference temperature at which sigma is set")
	AddSetting("T_init",zonal=T, 	comment="Initial temperature field")
	AddSetting("cp_h",				comment="specific heat for heavy phase")
	AddSetting("cp_l",				comment="specific heat for light phase")
	AddSetting("k_h", 				comment="thermal conductivity for heavy phase")
	AddSetting("k_l", 				comment="thermal conductivity for light phase")
	AddSetting("dT",				comment="Application of vertical temp gradient to speed up initialisation")
	AddGlobal("TempChange")
	if (Options$planarBenchmark){
		AddSetting("T_c", default="10")
		AddSetting("T_h", default="20")
		AddSetting("T_0", default="4")
		AddSetting("myL", default="100")
		AddSetting("MIDPOINT", default="51")
		AddSetting("PLUSMINUS", default="1")
		AddNodeType("BWall",group="ADDITIONALS")
		AddNodeType("TWall",group="ADDITIONALS")
	}
	AddDensity("RK1", dx=0, dy=0, dz=0, group="Thermal")
	AddDensity("RK2", dx=0, dy=0, dz=0, group="Thermal")
	AddDensity("RK3", dx=0, dy=0, dz=0, group="Thermal")
	AddField("RK1",stencil3d=1, group="Thermal")
	AddField("RK2",stencil3d=1, group="Thermal")
	AddField("RK3",stencil3d=1, group="Thermal")

	AddStage("CopyDistributions", "TempCopy",  save=Fields$group %in% c("g","h","Vel","nw", "PF","Thermal"))
	AddStage("RK_1", "TempUpdate1", save=Fields$name=="RK1", load=DensityAll$group=="Vel")
	AddStage("RK_2", "TempUpdate2", save=Fields$name=="RK2", load=DensityAll$name %in% c("U","V","W","RK1"))
	AddStage("RK_3", "TempUpdate3", save=Fields$name=="RK3", load=DensityAll$name %in% c("U","V","W","RK1","RK2"))
	AddStage("RK_4", "TempUpdate4", save=Fields$name %in% c("Temp","SurfaceTension"), load=DensityAll$name %in% c("U","V","W","RK1","RK2","RK3"))
	AddAction("TempToSteadyState", c("CopyDistributions","RK_1", "RK_2", "RK_3", "RK_4"))	
}

# Stages - processes to run for initialisation and each iteration
AddStage("WallInit"  , "Init_wallNorm", save=Fields$group=="nw")
AddStage("calcWall"  , "calcWallPhase", save=Fields$name=="PhaseF", load=DensityAll$group=="nw")

if (Options$OutFlow){
	AddStage("PhaseInit" , "Init", save=Fields$name=="PhaseF")
    AddStage("BaseInit"  , "Init_distributions", save=Fields$group %in% c("g","h","Vel","gold","hold","PF"))
    AddStage("calcPhase" , "calcPhaseF",	 save=Fields$name=="PhaseF", 
                                                 load=DensityAll$group %in% c("g","h","Vel","gold","hold","nw"))
    AddStage("BaseIter"  , "Run"       ,         save=Fields$group %in% c("g","h","Vel","nw","gold","hold","nw"), 
	                                         load=DensityAll$group %in% c("g","h","Vel","nw","gold","hold","nw"))
} else if (Options$thermo){
	AddStage("PhaseInit" , "Init", save=Fields$group %in% c("PF","Thermal") )
    AddStage("BaseInit"  , "Init_distributions", save=Fields$group %in% c("g","h","Vel","PF"))
    AddStage("calcPhase" , "calcPhaseF",	     save=Fields$name=="PhaseF", 
					                             load=DensityAll$group %in% c("g","h","Vel","nw") )
    AddStage("BaseIter"  , "Run"       ,         save=Fields$group %in% c("g","h","Vel","nw","Thermal"), 
	                                	         load=DensityAll$group %in% c("g","h","Vel","nw","Thermal","PF"))
} else {
	AddStage("PhaseInit" , "Init", save=Fields$name=="PhaseF")
    AddStage("BaseInit"  , "Init_distributions", save=Fields$group %in% c("g","h","Vel","PF"))
    AddStage("calcPhase" , "calcPhaseF",	 save=Fields$name=="PhaseF", 
					         load=DensityAll$group %in% c("g","h","Vel","nw") )
    AddStage("BaseIter"  , "Run"       ,         save=Fields$group %in% c("g","h","Vel","nw"), 
	                                	 load=DensityAll$group %in% c("g","h","Vel","nw"))
}
if (Options$thermo){
	AddAction("Iteration", c("BaseIter", "calcPhase", "calcWall","RK_1", "RK_2", "RK_3", "RK_4"))
	AddAction("Init"     , c("PhaseInit","WallInit" , "calcWall","BaseInit"))
} else {
	AddAction("Iteration", c("BaseIter", "calcPhase", "calcWall"))
	AddAction("Init"     , c("PhaseInit","WallInit" , "calcWall","BaseInit"))
}

# 	Outputs:
AddQuantity(name="Rho",unit="kg/m3")
AddQuantity(name="PhaseField",unit="1")
AddQuantity(name="U",	  unit="m/s",vector=T)
AddQuantity(name="P",	  unit="Pa")
AddQuantity(name="Normal", unit=1, vector=T)
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

AddSetting(name="ContactAngle", radAngle='ContactAngle*3.1415926535897/180', default='90', comment='Contact angle in degrees')
AddSetting(name='radAngle', comment='Conversion to rads for calcs')

#Domain initialisation (pre-defined set-ups)
AddSetting(name="RTI_Characteristic_Length", default=-999, comment='Use for RTI instability')

AddSetting(name="Radius", default="0.0", comment='Diffuse Sphere Radius')
AddSetting(name="CenterX", default="0.0", comment='Diffuse sphere center_x')
AddSetting(name="CenterY", default="0.0", comment='Diffuse sphere center_y')
AddSetting(name="CenterZ", default="0.0", comment='Diffuse sphere center_z')
AddSetting(name="BubbleType",default="1.0", comment='droplet(1.0) or bubble(-1.0)?!')

AddSetting(name="DonutTime", default="0.0", comment='Radius of a Torus - initialised to travel along x-axis')
AddSetting(name="Donut_h",   default="0.0", comment='Half donut thickness, i.e. the radius of the cross-section')
AddSetting(name="Donut_D",   default="0.0", comment='Dilation factor along the x-axis')
AddSetting(name="Donut_x0",  default="0.0", comment='Position along x-axis')

# 	Inputs: Fluid Properties
AddSetting(name="tau_l", comment='relaxation time (low density fluid)')
AddSetting(name="tau_h", comment='relaxation time (high density fluid)')
AddSetting(name="Viscosity_l", tau_l='(3*Viscosity_l)', default=0.16666666, comment='kinematic viscosity')
AddSetting(name="Viscosity_h", tau_h='(3*Viscosity_h)', default=0.16666666, comment='kinematic viscosity')
#	Inputs: Flow Properties
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

# Velocity Tracking on Centerline:
AddSetting(name="xyzTrack", default=1,comment='x<-1, y<-2, z<-3')
AddNodeType("Centerline",group="ADDITIONALS")
# Allow for smoothing of sharp interface initiation by diffusion
AddNodeType("Smoothing",group="ADDITIONALS")
#  For RTI interface tracking
AddNodeType("Spiketrack",group="ADDITIONALS")
AddNodeType("Saddletrack",group="ADDITIONALS")
AddNodeType("Bubbletrack",group="ADDITIONALS")

AddNodeType(name="MovingWall_N", group="BOUNDARY")
AddNodeType(name="MovingWall_S", group="BOUNDARY")
AddNodeType(name="NVelocity", group="BOUNDARY")
if (Options$OutFlow){
AddNodeType(name="ENeumann", group="BOUNDARY")
AddNodeType(name="EConvect", group="BOUNDARY")
}
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
AddGlobal(name="KineticEnergy",comment='Measure of kinetic energy', unit="J")

AddGlobal(name="GasTotalVelocity", comment='use to determine avg velocity of bubbles', unit="m/s")
AddGlobal(name="GasTotalVelocityX", comment='use to determine avg velocity of bubbles', unit="m/s")
AddGlobal(name="GasTotalVelocityY", comment='use to determine avg velocity of bubbles', unit="m/s")
AddGlobal(name="GasTotalVelocityZ", comment='use to determine avg velocity of bubbles', unit="m/s")
AddGlobal(name="GasCells",	   comment='use in line with GasTotalVelocity to determine average velocity', unit="1")

AddGlobal(name="LiqTotalVelocity", 	comment='use to determine avg velocity of droplets', unit="m/s")
AddGlobal(name="LiqTotalVelocityX", comment='use to determine avg velocity of droplets', unit="m/s")
AddGlobal(name="LiqTotalVelocityY", comment='use to determine avg velocity of droplets', unit="m/s")
AddGlobal(name="LiqTotalVelocityZ", comment='use to determine avg velocity of droplets', unit="m/s")
AddGlobal(name="LiqCells",	   		comment='use in line with LiqTotalVelocity to determine average velocity', unit="1")
AddGlobal(name="DropFront",	op="MAX",  comment='Highest location of droplet', unit="m")