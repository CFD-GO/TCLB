x = c(0,1,-1);
P = expand.grid(x=0:2,y=0:2,z=0:2)
U = expand.grid(x,x,x)

fname = paste("f",P$x,P$y,P$z,sep="")
AddDensity(
	name = fname,
	dx   = U[,1],
	dy   = U[,2],
	dz   = U[,3],
	comment=paste("density F",1:27-1),
	group="f"
)

AddDensity( name="fx",  group="Force", parameter=TRUE)
AddDensity( name="fy",  group="Force", parameter=TRUE)
AddDensity( name="fz",  group="Force", parameter=TRUE)
AddDensity( name="sol", group="Force", parameter=TRUE)


AddQuantity(name="P",unit="Pa")
AddQuantity(name="U",unit="m/s",vector=T)

AddQuantity(name="Solid",unit="1")
AddQuantity(name="F",unit="N/m3",vector=T)

AddSetting(name="nu", default=0.16666666, comment='Viscosity')
AddSetting(name="nubuffer",default=0.01, comment='Viscosity in the buffer layer')
AddSetting(name="Velocity", default="0m/s", comment='Inlet velocity', zonal=TRUE)
AddSetting(name="Pressure", default="0Pa", comment='Inlet pressure', zonal=TRUE)
AddSetting(name="Turbulence", comment='Turbulence intensity', zonal=TRUE)
AddSetting(name="GalileanCorrection",default=1.,comment='Galilean correction term')
AddSetting(name="ForceX", default=0, comment='Force force X')
AddSetting(name="ForceY", default=0, comment='Force force Y')
AddSetting(name="ForceZ", default=0, comment='Force force Z')
AddSetting(name="Smag", default=0, comment='Smagorinsky coefficient for SGS modeling')
AddSetting(name="Omega", default=1, comment='relaxation rate for 3rd order cumulants')

AddSetting(name="ParticleVelocityLimit", default=0.05, comment='(local) particle velocity limit', unit="1m/s")

AddGlobal(name="Flux", comment='Volume flux', unit="m3/s")
AddGlobal(name="Drag", comment='Force exerted on body in X-direction', unit="N")
AddGlobal(name="Lift", comment='Force exerted on body in Z-direction', unit="N")
AddGlobal(name="Lateral", comment='Force exerted on body in Y-direction', unit="N")

AddNodeType(name="WVelocityTurbulent", group="BOUNDARY")
AddNodeType(name="NVelocity", group="BOUNDARY")
AddNodeType(name="SVelocity", group="BOUNDARY")
AddNodeType(name="NPressure", group="BOUNDARY")
AddNodeType(name="SPressure", group="BOUNDARY")
AddNodeType(name="NSymmetry", group="ADDITIONALS")
AddNodeType(name="SSymmetry", group="ADDITIONALS")
AddNodeType(name="Body", group="BODY")
AddNodeType(name="IB", group="HO_BOUNDARY")

for (f in fname) AddField(f,dx=0,dy=0,dz=0) # Make f accessible also in present node (not only streamed)

#Averaging values
if (Options$AVG) {
AddQuantity(name="KinE",comment="Turbulent kinetic energy")
AddQuantity(name="ReStr",comment="Reynolds stress off-diagonal component",vector=T)
AddQuantity(name="Dissipation",comment="Dissipation e")
AddQuantity(name="avgU",unit="m/s",vector=T)
AddQuantity(name="varU",vector=T)
AddQuantity(name="averageP",unit="Pa")

AddDensity(name="avgP",dx=0,dy=0,dz=0,average=TRUE)
AddDensity(name="varUX",dx=0,dy=0,dz=0,average=TRUE)
AddDensity(name="varUY",dx=0,dy=0,dz=0,average=TRUE)
AddDensity(name="varUZ",dx=0,dy=0,dz=0,average=TRUE)
AddDensity(name="varUXUY",dx=0,dy=0,dz=0,average=TRUE)
AddDensity(name="varUXUZ",dx=0,dy=0,dz=0,average=TRUE)
AddDensity(name="varUYUZ",dx=0,dy=0,dz=0,average=TRUE)
AddDensity(name="avgdxu2",dx=0,dy=0,dz=0,average=TRUE)
AddDensity(name="avgdyv2",dx=0,dy=0,dz=0,average=TRUE)
AddDensity(name="avgdzw2",dx=0,dy=0,dz=0,average=TRUE)
AddDensity(name="avgUX",average=TRUE)
AddDensity(name="avgUY",average=TRUE)
AddDensity(name="avgUZ",average=TRUE)

AddField(name="avgUX",dx=c(-1,1),average=TRUE)
AddField(name="avgUY",dy=c(-1,1),average=TRUE)
AddField(name="avgUZ",dz=c(1,-1),average=TRUE)
}


AddStage("BaseIteration", "Run", save=Fields$group %in% c("f","Force"), load = DensityAll$group %in% c("f","Force"))
AddStage("CalcF", save=Fields$group == "Force", load = DensityAll$group %in% c("f","Force"), particle=TRUE)

AddAction("Iteration", c("BaseIteration", "CalcF"))
AddAction("Init", c("BaseInit", "CalcF"))

AddNodeType(name="EPressure", group="BOUNDARY")
AddNodeType(name="EVelocity", group="BOUNDARY")
AddNodeType(name="Wall", group="BOUNDARY")
AddNodeType(name="WPressure", group="BOUNDARY")
AddNodeType(name="WVelocity", group="BOUNDARY")
AddNodeType(name="MRT", group="COLLISION")
