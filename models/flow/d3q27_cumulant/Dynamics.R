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

AddQuantity( name="P",unit="Pa")
AddQuantity( name="U",unit="m/s",vector=T)
AddQuantity( name="M", unit="N", vector=T)

AddSetting(name="nu", default=0.16666666, comment='Viscosity')
AddSetting(name="nubuffer",default=0.01, comment='Viscosity in the buffer layer')
AddSetting(name="Velocity", default="0m/s", comment='Inlet velocity', zonal=TRUE)
AddSetting(name="Pressure", default="0Pa", comment='Inlet pressure', zonal=TRUE)
AddSetting(name="Turbulence", comment='Turbulence intensity', zonal=TRUE)

AddSetting(name="GalileanCorrection",default=1.,comment='Galilean correction term')
AddSetting(name="ForceX",default=0, comment='Force force X')
AddSetting(name="ForceY",default=0,comment='Force force Y')
AddSetting(name="ForceZ",default=0,comment='Force force Z')
AddSetting(name="Smag", default=0, comment='Smagorinsky coefficient for SGS modeling')
AddSetting(name="Omega", default=1, comment='relaxation rate for 3rd order cumulants')

AddGlobal(name="Flux", comment='Volume flux', unit="m3/s")
AddGlobal(name="Drag", comment='Force exerted on body in X-direction', unit="N")
AddGlobal(name="Lift", comment='Force exerted on body in Z-direction', unit="N")
AddGlobal(name="Lateral", comment='Force exerted on body in Y-direction', unit="N")

AddNodeType("WVelocityTurbulent","BOUNDARY")
AddNodeType("NVelocity", "BOUNDARY")
AddNodeType("SVelocity", "BOUNDARY")
AddNodeType("NPressure", "BOUNDARY")
AddNodeType("SPressure", "BOUNDARY")


AddNodeType("NSymmetry", "ADDITIONALS")
AddNodeType("SSymmetry", "ADDITIONALS")
AddNodeType("Body", "BODY")
	
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
