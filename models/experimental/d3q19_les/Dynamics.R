

MRTMAT = matrix(c(
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
-30,-11,-11,-11,-11,-11,-11,8,8,8,8,8,8,8,8,8,8,8,8,
12,-4,-4,-4,-4,-4,-4,1,1,1,1,1,1,1,1,1,1,1,1,
0,1,-1,0,0,0,0,1,-1,1,-1,1,-1,1,-1,0,0,0,0,
0,-4,4,0,0,0,0,1,-1,1,-1,1,-1,1,-1,0,0,0,0,
0,0,0,1,-1,0,0,1,1,-1,-1,0,0,0,0,1,-1,1,-1,
0,0,0,-4,4,0,0,1,1,-1,-1,0,0,0,0,1,-1,1,-1,
0,0,0,0,0,1,-1,0,0,0,0,1,1,-1,-1,1,1,-1,-1,
0,0,0,0,0,-4,4,0,0,0,0,1,1,-1,-1,1,1,-1,-1,
 0,2,2,-1,-1,-1,-1,1,1,1,1,1,1,1,1,-2,-2,-2,-2,
0,-4,-4,2,2,2,2,1,1,1,1,1,1,1,1,-2,-2,-2,-2,
 0,0,0,1,1,-1,-1,1,1,1,1,-1,-1,-1,-1,0,0,0,0,
0,0,0,-2,-2,2,2,1,1,1,1,-1,-1,-1,-1,0,0,0,0,
 0,0,0,0,0,0,0,1,-1,-1,1,0,0,0,0,0,0,0,0,
 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,-1,-1,1,
 0,0,0,0,0,0,0,0,0,0,0,1,-1,-1,1,0,0,0,0,
0,0,0,0,0,0,0,1,-1,1,-1,-1,1,-1,1,0,0,0,0,
0,0,0,0,0,0,0,-1,-1,1,1,0,0,0,0,1,-1,1,-1,
0,0,0,0,0,0,0,0,0,0,0,1,1,-1,-1,-1,-1,1,1
),19,19)

v = diag(t(MRTMAT) %*% MRTMAT)
MRTMAT.inv = diag(1/v) %*% t(MRTMAT)

selU = c(4,6,8)

AddDensity(
	name = paste("f",0:18,sep=""),
	dx   = MRTMAT[,selU[1]],
	dy   = MRTMAT[,selU[2]],
	dz   = MRTMAT[,selU[3]],
	comment=paste("density F",0:18),
	group="f"
)


AddDensity(
	name = "w",
	comment="weight fluid-solid",
	group="w",
	parameter=T
)

AddQuantity( name="Rho",unit="kg/m3")
AddQuantity( name="U",unit="m/s",vector=T)
# AddQuantity( name="W")
AddQuantity( name="WB",adjoint=T)

# AddSetting(name="omega", comment='one over relaxation time')
AddSetting(name="nu", default=0.16666666, comment='viscosity')
AddSetting(name="Velocity", default="0m/s", comment='inlet velocity', zonal=T)
AddSetting(name="Density", default=1, comment='inlet density', zonal=T)
AddSetting(name="Theta", default=1, comment='inlet density')
AddSetting(name="Turbulence", default=0, comment='amount of turbulence in init and on inlet', zonal=T)

AddSetting(name="ForceX", default="0N", comment='Force[x]')
AddSetting(name="ForceY", default="0N", comment='Force[y]')
AddSetting(name="ForceZ", default="0N", comment='Force[z]')

AddGlobal(name="Flux", comment='pressure loss')
AddGlobal(name="EnergyFlux", comment='pressure loss')
AddGlobal(name="PressureFlux", comment='pressure loss')
AddGlobal(name="PressureDiff", comment='pressure loss')
AddGlobal(name="MaterialPenalty", comment='quadratic penalty for intermediate material parameter')

AddSetting(name="Smag", default=0, comment='Smagorynsky constant')



