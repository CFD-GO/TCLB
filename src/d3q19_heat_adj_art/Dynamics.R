
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
	name = paste("T",1:7-1,sep=""),
	dx   = d3q7[,1],
	dy   = d3q7[,2],
	dz   = d3q7[,3],
	comment=paste("heat LB density G",1:7-1),
	group="g"
)

AddDensity(
	name = "w",
	comment="weight fluid-solid",
	group="w",
	parameter=T
)

AddQuantity( name="W")
AddQuantity( name="WB",adjoint=T)

AddQuantity( name="Rho" )
AddQuantity( name="T" )
AddQuantity( name="U", vector=T )

AddSetting(name="nu", default=0.16666666, comment='viscosity')
AddSetting(name="Velocity", default="0m/s", comment='inlet velocity', zonal=TRUE)
AddSetting(name="Pressure", default="0Pa", comment='inlet pressure', zonal=TRUE)
AddSetting(name="Temperature", default=1, comment='inlet density', zonal=TRUE)
AddSetting(name="LimitTemperature", default=1, comment='inlet density', zonal=TRUE)
AddSetting(name="FluidAlpha", default=1, comment='inlet density')
AddSetting(name="SolidAlpha", comment='Heat conductivity of solid')
AddSetting(name="Buoyancy", comment='Buoyancy coefficient of temperature')

AddSetting(name="PorocityGamma", comment='Gamma in hiperbolic transformation of porocity (-infty,1)')
AddSetting(name="PorocityTheta", comment='Theta in hiperbolic transformation of porocity', PorocityGamma='1.0 - exp(PorocityTheta)')

AddGlobal(name="HeatFlux", comment='Flux of heat', unit="Km3/s")
AddGlobal(name="HeatSquareFlux", comment='Flux of temperature squered', unit="K2m3/s")
AddGlobal(name="Flux", comment='Volume flux', unit="m3/s")
AddGlobal(name="TemperatureAtPoint", comment='Integral of temperature', unit="K")
AddGlobal(name="HighTemperature", comment='Penalty for high temperature')
AddGlobal(name="LowTemperature", comment='Penalty for low temperature')
AddGlobal(name="MaterialPenalty", comment='Quadratic penalty for intermediate material parameter', unit="m3")

AddNodeType("Heater","ADDITIONALS")
AddNodeType("HeatSource","ADDITIONALS")
AddNodeType("Thermometer","OBJECTIVE")
