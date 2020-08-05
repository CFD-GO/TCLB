source("lib/lattice.R")

# declaration of lattice (velocity) directions
x = c(0,1,-1);
P = expand.grid(x=0:2,y=0:2,z=0:2)
U = expand.grid(x,x,x)

# declaration of densities
fname = paste("f",P$x,P$y,P$z,sep="")
AddDensity(
	name = fname,
	dx   = U[,1],
	dy   = U[,2],
	dz   = U[,3],
	comment=paste("flow LB density F",1:27-1),
	group="f"
)


mom_d3q7 = provideDimnames(d3q7, base=list(paste(c(0:7)), c("x", "y", "z")))
mom_d3q7 <- replace(mom_d3q7, mom_d3q7 == -1, 2) 
mom_d3q7 <- as.data.frame(mom_d3q7)
hname = paste("h",mom_d3q7$x,mom_d3q7$y,mom_d3q7$z,sep="")

AddDensity(
        name = hname,
        dx   = d3q7[,1],
        dy   = d3q7[,2],
        dz   = d3q7[,3],
        comment=paste("heat LB density H",1:7-1),
        group="h"
)

#	Inputs: Flow Properties
AddSetting(name="VelocityX", 	default="0m/s",		comment='inlet/outlet/init x-velocity component', zonal=TRUE)
AddSetting(name="VelocityY", 	default="0m/s",	 	comment='inlet/outlet/init y-velocity component', zonal=TRUE)
AddSetting(name="VelocityZ", 	default="0m/s", 	comment='inlet/outlet/init z-velocity component', zonal=TRUE)
AddSetting(name="Pressure" , 	default="0Pa",  	comment='inlet/outlet/init pressure', zonal=TRUE)
AddSetting(name="GravitationX", default=0.0, 		comment='applied rho*GravitationX')
AddSetting(name="GravitationY", default=0.0,	 	comment='applied rho*GravitationY')
AddSetting(name="GravitationZ", default=0.0,	 	comment='applied rho*GravitationZ')
AddSetting(name="nu",		    default=0.16666666,	comment='kinematic viscosity')


#	Inputs: CFD Enhancements ;-)
AddSetting(name="GalileanCorrection",		default=1.,		comment='Galilean correction term')
AddSetting(name="nu_buffer",				default=0.01, 	comment='kinematic viscosity in the buffer layer')
AddSetting(name="conductivity_buffer",		default=0.01, 	comment='thermal conductivity in the buffer layer')
AddSetting(name="Omegafor3rdCumulants", 	default=1, 		comment='relaxation rate for 3rd order cumulants')
AddSetting(name="h_stability_enhancement",  default=1.0, 	comment='magic stability enhancement')


# 	Inputs: General Thermal Properties
AddSetting(name="InitTemperature", 		default=0, comment='Initial/Inflow temperature distribution', 	zonal=T)
AddSetting(name="InitHeatFlux", 		default=0, comment='Initial/Inflow heat flux through boundary', zonal=T)


# 	Inputs: Fluid Thermal Properties
AddSetting(name="conductivity", 		default=0.16666666, comment='thermal conductivity of fluid (W/(m·K))',	zonal=T)
AddSetting(name="material_density", 	default=1.0, 		comment='density of material [kg/m3]', 				zonal=T)
AddSetting(name="cp", 					default=1.0, 		comment='specific heat capacity at constant pressure of fluid (J/(kg·K))', zonal=T)
AddSetting(name="BoussinesqCoeff", 		default=1.0, 		comment='BoussinesqCoeff=rho_0*thermal_exp_coeff')


#	Globals - table of global integrals that can be monitored and optimized
AddGlobal(name="FDrag",    		comment='Force exerted on body in X-direction', unit="N")
AddGlobal(name="FLateral", 		comment='Force exerted on body in Y-direction', unit="N")
AddGlobal(name="FLift",    		comment='Force exerted on body in Z-direction', unit="N")

AddGlobal(name="XHydroFLux",	comment='Momentum flux in X-direction', unit="kg/s")
AddGlobal(name="YHydroFLux",    comment='Momentum flux in Y-direction', unit="kg/s")
AddGlobal(name="ZHydroFLux",    comment='Momentum flux in Z-direction', unit="kg/s")
AddGlobal(name="XHydroFLux2",   comment='Momentum flux (2nd logger) in X-direction', unit="kg/s")
AddGlobal(name="YHydroFLux2",   comment='Momentum flux (2nd logger) in Y-direction', unit="kg/s")
AddGlobal(name="ZHydroFLux2",   comment='Momentum flux (2nd logger) in Z-direction', unit="kg/s")

AddGlobal(name="HeatFluxX",     comment='Heat flux in X-direction', unit="W")
AddGlobal(name="HeatFluxY",     comment='Heat flux in Y-direction', unit="W")
AddGlobal(name="HeatFluxZ",     comment='Heat flux in Z-direction', unit="W")
AddGlobal(name="HeatFluxX2",    comment='Heat flux (2nd logger) in X-direction', unit="W")
AddGlobal(name="HeatFluxY2",    comment='Heat flux (2nd logger) in Y-direction', unit="W")
AddGlobal(name="HeatFluxZ2",    comment='Heat flux (2nd logger) in Z-direction', unit="W")

AddGlobal(name="HeatSource",   comment='Total Heat flux from body', unit="W")


# 	Outputs:
AddQuantity(name="Rho", 	unit="kg/m3")
AddQuantity(name="U", 		unit="m/s",vector=T )
AddQuantity(name="H", 		unit="J" )
AddQuantity(name="T", 		unit="K")
# 	Debug-Outputs:
AddQuantity(name="m00_F" )
AddQuantity(name="material_density", unit="kg/m3" )
AddQuantity(name="cp", 				 unit="J/kg/K")
AddQuantity(name="conductivity", 	 unit="W/m/K" )
AddQuantity(name="RawU", 			 unit="m/s", vector=T )


#	Boundary things
AddNodeType(name="ForceMeasurmentZone", 	group="OBJECTIVEFORCE")
AddNodeType(name="FluxMeasurmentZone1", group="OBJECTIVEFLUX")
AddNodeType(name="FluxMeasurmentZone2", group="OBJECTIVEFLUX")

AddNodeType(name="DarcySolid", group="ADDITIONALS")
AddNodeType(name="Smoothing", group="ADDITIONALS")

AddNodeType(name="HeaterDirichletTemperatureEQ", group="ADDITIONALS_HEAT")
AddNodeType(name="HeaterDirichletTemperatureABB", group="ADDITIONALS_HEAT")
AddNodeType(name="HeaterSource", 				  group="ADDITIONALS_HEAT")
AddNodeType(name="HeaterNeumannHeatFluxCylinder", group="ADDITIONALS_HEAT")
AddNodeType(name="HeaterNeumannHeatFluxEast", 	  group="ADDITIONALS_HEAT")

AddNodeType(name="CM", 						group="COLLISION")
#AddNodeType(name="CM_HIGHER", 				group="COLLISION")
AddNodeType(name="CM_NONLINEAR", 			group="COLLISION")
#AddNodeType(name="Cumulants", 				group="COLLISION")


#	Benchmark things
AddSetting(name="CylinderCenterX", 		default="0", comment='X coord of cylinder with imposed heat flux')
AddSetting(name="CylinderCenterY", 		default="0", comment='Y coord of cylinder with imposed heat flux')

AddSetting(name="CylinderCenterX_GH",	default="0", comment='X coord of Gaussian Hill')
AddSetting(name="CylinderCenterY_GH",	default="0", comment='Y coord of Gaussian Hill')
AddSetting(name="Sigma_GH", 		 	default="1", comment='Initial width of the Gaussian Hill', zonal=T)

########## OPTIONAL COMPILATION ##########

#	Interpolated BounceBack Node
if(Options$IBB){
	for (f in fname) AddField(f,dx=0,dy=0,dz=0) # Make f accessible also in present node (not only streamed)
	for (h in hname) AddField(h,dx=0,dy=0,dz=0) # Make h accessible also in present node (not only streamed)
	
	AddNodeType(name="HeaterDirichletTemperatureIABB", group="HO_BOUNDARY_HEAT") 
	AddNodeType("ThermalIBB", 					  group="HO_BOUNDARY_HEAT") 
	AddNodeType("HydroIBB", 					  group="HO_BOUNDARY_HYDRO") 
}

#	Smagorinsky coefficient
if(Options$SMAG)
{
	AddSetting(name="Smag", default=0, comment='Smagorinsky coefficient for SGS modeling')
}

AddDensity(name="U", dx=0, dy=0, dz=0, group="Vel")  
# AddDensity(name="V", dx=0, dy=0, dz=0, group="Vel")
# AddDensity(name="W", dx=0, dy=0, dz=0, group="Vel")
if (Options$OutFlowConvective)
{
	holdname = paste("hold",mom_d3q7$x,mom_d3q7$y,mom_d3q7$z,sep="")
	AddDensity(
		name = holdname,
		dx   = 0,
		dy   = 0,
		dz   = 0,
		comment=paste("heat LB density H",1:7-1),
		group="hold"
	)

	foldname =  paste("fold",P$x,P$y,P$z,sep="")
	AddDensity(
		name = foldname,
		dx   = 0,
		dy   = 0,
		dz   = 0,
		comment=paste("flow LB density F",1:27-1),
		group="fold"
	)

	for (d in rows(DensityAll)) {
		AddField( name=d$name, dx=-d$dx-1, dy=-d$dy, dz=-d$dz )
	}

	AddField(name="U",dx=c(-1,0,0))
	AddNodeType(name="EConvective", group="BOUNDARY")
}

if (Options$OutFlowNeumann)
{
	for (d in rows(DensityAll)) {
		AddField(name=d$name, dx=-d$dx-1, dy=-d$dy, dz=-d$dz )
	}
	AddNodeType(name="ENeumann", group="BOUNDARY")
}


#	Averaging values
if (Options$AVG) {
	AddQuantity(name="KinE",		comment="Turbulent kinetic energy")
	AddQuantity(name="ReStr",		comment="Reynolds stress off-diagonal component",vector=T)
	AddQuantity(name="Dissipation",	comment="Dissipation e")
	AddQuantity(name="averageU",	unit="m/s",vector=T)
	AddQuantity(name="varU",		vector=T)
	AddQuantity(name="averageP",	unit="Pa")
	AddQuantity(name="averageT",	unit="K")

	AddDensity(name="avgT",		dx=0,dy=0,dz=0,average=TRUE)
	AddDensity(name="avgP",		dx=0,dy=0,dz=0,average=TRUE)
	AddDensity(name="varUX",	dx=0,dy=0,dz=0,average=TRUE)
	AddDensity(name="varUY",	dx=0,dy=0,dz=0,average=TRUE)
	AddDensity(name="varUZ",	dx=0,dy=0,dz=0,average=TRUE)
	AddDensity(name="varUXUY",	dx=0,dy=0,dz=0,average=TRUE)
	AddDensity(name="varUXUZ",	dx=0,dy=0,dz=0,average=TRUE)
	AddDensity(name="varUYUZ",	dx=0,dy=0,dz=0,average=TRUE)
	AddDensity(name="avgdxu2",	dx=0,dy=0,dz=0,average=TRUE)
	AddDensity(name="avgdyv2",	dx=0,dy=0,dz=0,average=TRUE)
	AddDensity(name="avgdzw2",	dx=0,dy=0,dz=0,average=TRUE)
	AddDensity(name="avgUX",	average=TRUE)
	AddDensity(name="avgUY",	average=TRUE)
	AddDensity(name="avgUZ",	average=TRUE)

	AddField(name="avgUX",		dx=c(-1,1),average=TRUE)
	AddField(name="avgUY",		dy=c(-1,1),average=TRUE)
	AddField(name="avgUZ",		dz=c(1,-1),average=TRUE)
}
AddNodeType(name="EPressure", group="BOUNDARY")
AddNodeType(name="EVelocity", group="BOUNDARY")
AddNodeType(name="Solid", group="BOUNDARY")
AddNodeType(name="Wall", group="BOUNDARY")
AddNodeType(name="WPressure", group="BOUNDARY")
AddNodeType(name="WVelocity", group="BOUNDARY")
AddNodeType(name="BGK", group="COLLISION")
AddNodeType(name="Body", group="BODY")
