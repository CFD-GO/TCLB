
Density = data.frame(
	name = paste("f[",0:8,"]"),
	dx   = c( 0, 1, 0,-1, 0, 1,-1,-1, 1),
	dy   = c( 0, 0, 1, 0,-1, 1, 1,-1,-1),
	dz   = c( 0, 0, 0, 0, 0, 0, 0, 0, 0),
	comment=paste("density F",0:8)
)

Quantities = data.frame(
	name = c("Rho","U"),
	type = c("type_f","type_v")
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

