source("lib/lattice.R")

AddDensity(
	name = paste("f",1:19-1,sep=""),
	dx   = d3q19[,1],
	dy   = d3q19[,2],
	dz   = d3q19[,3],
	comment=paste("flow LB density F",1:19-1),
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

AddQuantity( name="Rho" )
AddQuantity( name="T" )
AddQuantity( name="U", vector=T )


AddSetting(name="nu", default=0.16666666, comment='viscosity')
AddSetting(name="Velocity", default="0m/s", comment='inlet velocity', zonal=TRUE)
AddSetting(name="Pressure", default="0Pa", comment='inlet pressure', zonal=TRUE)
AddSetting(name="Temperature", default=1, comment='inlet density', zonal=TRUE)
AddSetting(name="FluidAlpha", default=1, comment='inlet density')

AddNodeType("Heater","ADDITIONALS")
