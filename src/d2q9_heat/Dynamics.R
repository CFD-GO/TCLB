
Density = data.frame(
	name = paste("f[",0:8,"]"),
	dx   = c( 0, 1, 0,-1, 0, 1,-1,-1, 1),
	dy   = c( 0, 0, 1, 0,-1, 1, 1,-1,-1),
	dz   = c( 0, 0, 0, 0, 0, 0, 0, 0, 0),
	comment=paste("density F",0:8)
)

Quantities = data.frame(
	name = c("Rho","U", "T"),
	type = c("type_f","type_v", "type_f")
)

f = PV(Density$name)

U = as.matrix(Density[,c("dx","dy")])

Density = rbind(Density, data.frame(
	name = paste("T[",0:8,"]"),
	dx   = c( 0, 1, 0,-1, 0, 1,-1,-1, 1),
	dy   = c( 0, 0, 1, 0,-1, 1, 1,-1,-1),
	dz   = c( 0, 0, 0, 0, 0, 0, 0, 0, 0),
	comment=paste("density T",0:8)
))

Settings = table_from_text("
        name                 derived                equation   comment
        omega                     NA                      NA   'one over relaxation time'
        nu                     omega      '1.0/(3*nu + 0.5)'   'viscosity'
        InletVelocity             NA                      NA   'inlet velocity'
        InletPressure   InletDensity   '1.0+InletPressure/3'   'inlet pressure'
        InletDensity              NA                      NA   'inlet density'
	InletTemperature          NA                      NA   'inlet temperature'
	InitTemperature           NA                      NA   'initial temperature'
	FluidAlfa                 NA                      NA   'thermal diffusivity of fluid'
")
