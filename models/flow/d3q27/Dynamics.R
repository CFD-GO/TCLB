
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
AddQuantity( name="Fd", unit="N", vector=T)

AddSetting(name="omega", comment='One over relaxation time')
AddSetting(name="nu", omega='1.0/(3*nu + 0.5)', default=0.16666666, comment='Viscosity')
AddSetting(name="Velocity", default="0m/s", comment='Inlet velocity', zonal=TRUE)
AddSetting(name="Pressure", default="0Pa", comment='Inlet pressure', zonal=TRUE)
AddSetting(name="Smag", comment='Smagorinsky constant')
AddSetting(name="Turbulence", comment='Turbulence intensity', zonal=TRUE)

AddSetting(name="ForceX", comment='Force force X')
AddSetting(name="ForceY", comment='Force force Y')
AddSetting(name="ForceZ", comment='Force force Z')

AddGlobal(name="XFlux", comment='Volume flux', unit="m3/s")
AddGlobal(name="YFlux", comment='Volume flux', unit="m3/s")
AddGlobal(name="ZFlux", comment='Volume flux', unit="m3/s")
AddGlobal(name="XDragForce", comment='Solid drag force', unit="N")
AddGlobal(name="YDragForce", comment='Solid drag force', unit="N")
AddGlobal(name="ZDragForce", comment='Solid drag force', unit="N")

AddNodeType(name="Smagorinsky", group="LES")
AddNodeType(name="Stab", group="ENTROPIC")
AddNodeType(name="NSymmetry", group="BOUNDARY")
AddNodeType(name="SSymmetry", group="BOUNDARY")
AddNodeType(name="ISymmetry", group="BOUNDARY")
AddNodeType(name="EPressure", group="BOUNDARY")
AddNodeType(name="Solid", group="BOUNDARY")
AddNodeType(name="Wall", group="BOUNDARY")
AddNodeType(name="WPressure", group="BOUNDARY")
AddNodeType(name="WVelocity", group="BOUNDARY")
AddNodeType(name="MRT", group="COLLISION")
