
Density = data.frame(
	name = paste("f",0:8,sep=""),
	dx   = c( 0, 1, 0,-1, 0, 1,-1,-1, 1),
	dy   = c( 0, 0, 1, 0,-1, 1, 1,-1,-1),
	dz   = c( 0, 0, 0, 0, 0, 0, 0, 0, 0),
	comment = paste("streamed density F",0:8),
	group = "f"
)

U = as.matrix(Density[,c("dx","dy")])

Density = rbind(Density,data.frame(
	name = paste("fs",0:8,sep=""),
	dx   = 0,
	dy   = 0,
	dz   = 0,
	comment = paste("density F",0:8),
	group = "fs"
))

Density = rbind(Density,data.frame(
	name = paste("phi",0:8,sep=""),
	dx   = c( 0, 1, 0,-1, 0, 1,-1,-1, 1),
	dy   = c( 0, 0, 1, 0,-1, 1, 1,-1,-1),
	dz   = c( 0, 0, 0, 0, 0, 0, 0, 0, 0),
	comment = paste("density F",0:8),
	group = "phi"
))

Quantities = data.frame(
	name = c("Rho", "U", "P", "F"),
	type = c("type_f", "type_v", "type_f", "type_v")
)

f  = PV(Density$name[Density$group=="f"])
fs = PV(Density$name[Density$group=="fs"])
ph = PV(Density$name[Density$group=="phi"])


Settings = table_from_text("
        name                 derived                equation   comment
        omega                     NA                      NA   'one over relaxation time'
        nu                     omega      '1.0/(3*nu + 0.5)'   'viscosity'
        InletVelocity             NA                      NA   'inlet velocity'
        OutletDensity             NA                      NA   'outlet density'
        InletDensity              NA                      NA   'inlet density'
        InitDensity              NA                      NA   'inlet density'
	WallDensity		NA			NA	'vapor/liquid density of wall'
	Temperature                 NA                      NA   'temperature of the liquid/gas'
	FAcc			NA			NA	'wtf'
	Magic			NA			NA	'wtf'
	MagicA			NA			NA	'wtf'
	MagicF			NA			NA	'wtf'
	GravitationY		NA			NA	'Gravitation in the direction of y'
	GravitationX		NA			NA	'Gravitation in the direction of x'
	MovingWallVelocity	NA			NA	'Velocity of the MovingWall'
	WetDensity		NA			NA	'wet density'
	DryDensity		NA			NA	'dry density'
")

Globals = table_from_text("
        name            in_objective   comment
        MovingWallForceX    1              'force x'
        MovingWallForceY    1              'force y'
")
