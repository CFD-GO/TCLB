	
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

AddSetting(name="nu", default=1.6666666, comment='Viscosity',zonal=TRUE)
AddSetting(name="Velocity", default="0m/s", comment='Inlet velocity', zonal=TRUE)
AddSetting(name="Pressure", default="0Pa", comment='Inlet pressure', zonal=TRUE)
AddSetting(name="Smag", comment='Smagorinsky constant')
AddSetting(name="Turbulence", comment='Turbulence intensity', zonal=TRUE)
AddSetting(name="Ave",default=0,comment='Averaging indicator') #Change default to 1,if you want for average values to be calculated,0 - for default,fast solution


AddSetting(name="GalileanCorrection",default=0.,comment='Galilean correction term')
AddSetting(name="ForceX", comment='Force force X')
AddSetting(name="ForceY", comment='Force force Y')
AddSetting(name="ForceZ", comment='Force force Z')

AddGlobal(name="Flux", comment='Volume flux', unit="m3/s")

AddNodeType("Smagorinsky", "LES")
AddNodeType("Stab", "ENTROPIC")
AddNodeType("SymmetryY", "BOUNDARY")
AddNodeType("SymmetryZ", "BOUNDARY")

#Adding terms for supporting time-correlation for synthetic turbulence

AddDensity( name="SynthTX",dx=0,dy=0,dz=0)
AddDensity( name="SynthTY",dx=0,dy=0,dz=0)
AddDensity( name="SynthTZ",dx=0,dy=0,dz=0)

#Averaging values
if (Ave == 1) {
AddQuantity( name="avgU",unit="m/s",vector=T)
AddQuantity( name="varU",comment="avgU",vector=T)
AddDensity( name="varUX",dx=0,dy=0,dz=0,average=T)
AddDensity( name="varUY",dx=0,dy=0,dz=0,average=T)
AddDensity( name="varUZ",dx=0,dy=0,dz=0,average=T)
AddDensity( name="avgUX",dx=0,dy=0,dz=0,average=T)
AddDensity( name="avgUY",dx=0,dy=0,dz=0,average=T)
AddDensity( name="avgUZ",dx=0,dy=0,dz=0,average=T)
}
