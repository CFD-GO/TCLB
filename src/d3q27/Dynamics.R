
U = expand.grid(-1:1,-1:1,-1:1)

AddDensity(
	name = paste("f",1:27-1,sep=""),
	dx   = U[,1],
	dy   = U[,2],
	dz   = U[,3],
	comment=paste("density F",1:27-1),
	group="f"
)

AddQuantity( name="P",unit="Pa")
AddQuantity( name="U",unit="m/s",vector=T)

AddSetting(name="omega", comment='One over relaxation time')
AddSetting(name="nu", omega='1.0/(3*nu + 0.5)', default=0.16666666, comment='Viscosity')
AddSetting(name="Velocity", default="0m/s", comment='Inlet velocity', zonal=TRUE)
AddSetting(name="Pressure", default="0Pa", comment='Inlet pressure', zonal=TRUE)
AddSetting(name="Smag", comment='Smagorinsky constant')
AddSetting(name="Turbulence", comment='Turbulence intensity', zonal=TRUE)

AddSetting(name="ForceX", comment='Force force X')
AddSetting(name="ForceY", comment='Force force Y')
AddSetting(name="ForceZ", comment='Force force Z')

AddGlobal(name="Flux", comment='Volume flux', unit="m3/s")

AddNodeType("Smagorinsky", "LES")
AddNodeType("Stab", "ENTROPIC")
AddNodeType("NSymmetry", "BOUNDARY")
AddNodeType("ISymmetry", "BOUNDARY")
