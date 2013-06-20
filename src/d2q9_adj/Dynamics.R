Density = data.frame(
	name = paste("f",0:8,sep=""),
	dx   = c( 0, 1, 0,-1, 0, 1,-1,-1, 1),
	dy   = c( 0, 0, 1, 0,-1, 1, 1,-1,-1),
	dz   = c( 0, 0, 0, 0, 0, 0, 0, 0, 0),
	group="f",
	comment=paste("density F",0:8)
)

Quantities = data.frame(
        name = c("Rho","U","RhoB","UB","W","WB"),
        type = c("real_t","vector_t","real_t","vector_t","real_t","real_t"),
	adjoint = c(F,F,T,T,F,T)
)

Settings = table_from_text("
        name                 derived                equation   comment
        omega                     NA                      NA   'one over relaxation time'
        nu                     omega      '1.0/(3*nu + 0.5)'   'viscosity'
        InletVelocity             NA                      NA   'inlet velocity'
        InletPressure   InletDensity   '1.0+InletPressure/3'   'inlet pressure'
        InletDensity              NA                      NA   'inlet density'
        InletTemperature          NA                      NA   'inlet temperature'
	HeaterTemperature	  NA			  NA   'temperature of the heater'
	LimitTemperature	  NA			  NA   'temperature of the heater'
	FluidAlpha		  NA			  NA   'heat conductivity of fluid'
	SolidAlpha		  NA			  NA   'heat conductivity of fluid'
	HeatSource		  NA			  NA   'heat conductivity of fluid'
	Inertia                   NA                      NA   'inertia of the transport equation'
")

Globals = table_from_text("
        name            in_objective   comment
        HeatFlux    1              'pressure loss'
        HeatSquareFlux    1              'pressure loss'
	PressDiff	1	'pressure difference'
        Flux    1              'pressure loss'
	Temperature 1 'integral of temperature'
	HighTemperature 1 'penalty for high temperature'
	LowTemperature  1 'penalty for low temperature'
")

f = PV(Density$name)
U = as.matrix(Density[,c("dx","dy")])

Density = rbind(Density, data.frame(
	name = "w",
	dx=0,dy=0,dz=0,
	group="w",
	comment = "Porocity"
))


