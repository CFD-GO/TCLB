
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

Density = rbind(Density, data.frame(
	name = paste("w"),
	dx   = c( 0),
	dy   = c( 0),
	dz   = c( 0),
	comment="porocity term"
))

DensityAD = Density

DensityAD$dx = -Density$dx
DensityAD$dy = -Density$dy
DensityAD$dz = -Density$dz
DensityAD$name = as.character(DensityAD$name);
i = grep("[[]", Density$name)
DensityAD$name[i] = sub("[[]","b[", Density$name[i])
DensityAD$name[-i] = paste(Density$name[-i], "b",sep="")

Density = rbind(Density,DensityAD)


Quantities = rbind(Quantities, data.frame(
        name = c("W", "WB"),
        type = c("type_f","type_f")
))



Settings = data.frame(
        name = c("omega","nu","UX_mid"),
        derived = c(NA,"omega",NA),
        equation = c(NA,"1.0/(3*nu + 0.5)",NA),
        comment = c("one over relaxation time", "viscosity", "velocity on inlet")
)
