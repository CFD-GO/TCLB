source('lib/lattice.R')
Ave = FALSE
x = c(0,1,-1);
P = expand.grid(x=0:2,y=0:2,z=0:2)
U = expand.grid(x,x,x)

AddDensity(
	name = paste("f",P$x,P$y,P$z,sep=""),
	dx   = U[,1],
	dy   = U[,2],
	dz   = U[,3],
	comment=paste("density F",1:27-1),
	group="f"
)

AddDensity(
        name = paste("g",1:7-1,sep=""),
        dx   = d3q7[,1],
        dy   = d3q7[,2],
        dz   = d3q7[,3],
        comment=paste("heat LB density G",1:7-1),
        group="g"
)

AddQuantity( name="P",unit="Pa")
AddQuantity( name="U",unit="m/s",vector=T)
AddQuantity( name="T",unit="K")

AddSetting(name="nu", default=0.16666666, comment='Viscosity')
AddSetting(name="nubuffer",default=0.01, comment='Viscosity in the buffer layer')
AddSetting(name="Velocity", default="0m/s", comment='Inlet velocity', zonal=TRUE)
AddSetting(name="Pressure", default="0Pa", comment='Inlet pressure', zonal=TRUE)
AddSetting(name="Turbulence", comment='Turbulence intensity', zonal=TRUE)
AddSetting(name="Temperature",comment='Temperature',zonal=TRUE)
AddSetting(name="Alpha",zonal=TRUE)
AddSetting(name="Buoyancy",unit="N/K")
AddSetting(name="BuoyancyT0",unit="K")


AddSetting(name="GalileanCorrection",default=0.,comment='Galilean correction term')
AddSetting(name="ForceX",default=0, comment='Force force X')
AddSetting(name="ForceY",default=0,comment='Force force Y')
AddSetting(name="ForceZ",default=0,comment='Force force Z')

AddGlobal(name="HeatFlux", comment='Heat flux', unit="Km3/s")

AddNodeType(name="WVelocityTurbulent", group="BOUNDARY")
AddNodeType(name="NSymmetry", group="BOUNDARY")
AddNodeType(name="SSymmetry", group="BOUNDARY")
AddNodeType(name="ISymmetry", group="BOUNDARY")
AddNodeType(name="OSymmetry", group="BOUNDARY")
AddNodeType(name="NVelocity", group="BOUNDARY")
AddNodeType(name="SVelocity", group="BOUNDARY")
AddNodeType(name="NPressure", group="BOUNDARY")
AddNodeType(name="SPressure", group="BOUNDARY")
AddNodeType(name="Heater", group="ADDITIONALS")
AddNodeType(name="SamplingPlane", group="ADDITIONALS")
	
#Adding terms for supporting time-correlation for synthetic turbulence


AddDensity( name="SynthTX",dx=0,dy=0,dz=0)
AddDensity( name="SynthTY",dx=0,dy=0,dz=0)
AddDensity( name="SynthTZ",dx=0,dy=0,dz=0)

#Averaging values
if (Ave) {

AddQuantity(name="KinE",comment="Turbulent kinetic energy")
AddQuantity( name="ReStr",comment="Reynolds stress off-diagonal component",vector=T)
AddQuantity( name="Dissipation",comment="Dissipation e")
AddQuantity( name="avgU",unit="m/s",vector=T)
AddQuantity( name="varU",vector=T)
AddQuantity( name="averageP",unit="Pa")
AddDensity( name="avgP",dx=0,dy=0,dz=0,average=T)
AddDensity( name="varUX",dx=0,dy=0,dz=0,average=T)
AddDensity( name="varUY",dx=0,dy=0,dz=0,average=T)
AddDensity( name="varUZ",dx=0,dy=0,dz=0,average=T)
AddDensity( name="varUXUY",dx=0,dy=0,dz=0,average=T)
AddDensity( name="varUXUZ",dx=0,dy=0,dz=0,average=T)
AddDensity( name="varUYUZ",dx=0,dy=0,dz=0,average=T)

AddDensity( name="avgdxu2",dx=0,dy=0,dz=0,average=T)
AddDensity( name="avgdyv2",dx=0,dy=0,dz=0,average=T)
AddDensity( name="avgdzw2",dx=0,dy=0,dz=0,average=T)
AddField(name="avgUX",dx=c(-1,1),average=T)
AddField(name="avgUY",dy=c(-1,1),average=T)
AddField(name="avgUZ",dz=c(1,-1),average=T)
}
AddNodeType(name="EPressure", group="BOUNDARY")
AddNodeType(name="EVelocity", group="BOUNDARY")
AddNodeType(name="Solid", group="BOUNDARY")
AddNodeType(name="Wall", group="BOUNDARY")
AddNodeType(name="WPressure", group="BOUNDARY")
AddNodeType(name="WVelocity", group="BOUNDARY")
AddNodeType(name="MRT", group="COLLISION")
