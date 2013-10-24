
MRT = function(x,y,z) {
  v = cbind(x,y,z)
  v2 = v
  v2[v2 == -1] = 2
  M = NULL
  for (i in 1:nrow(v2)) {
    h = apply(t(v) ** v2[i,], 2, prod)
    M = rbind(M,h)
  }
  M
}

MRTMAT = MRT( x=c(0,1,-1,0,0,0,0,1,-1,1,-1,1,-1,1,-1,0,0,0,0),
              y=c(0,0,0,1,-1,0,0,1,1,-1,-1,0,0,0,0,1,-1,1,-1),
              z=c(0,0,0,0,0,1,-1,0,0,0,0,1,1,-1,-1,1,1,-1,-1))
MTT    = MRT( x=c(0,1,-1,0,0,0,0),
              y=c(0,0,0,1,-1,0,0),
              z=c(0,0,0,0,0,1,-1))

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

MRTT = matrix(c(
1, 0, 0, 0, 0, 0, 0,
1, 1, 0, 0, 1, 0, 0,
1,-1, 0, 0, 1, 0, 0,
1, 0, 1, 0, 0, 1, 0,
1, 0,-1, 0, 0, 1, 0,
1, 0, 0, 1, 0, 0, 1, 
1, 0, 0,-1, 0, 0, 1),7,7)

AddDensity(
	name = paste("f",0:18,sep=""),
	dx   = MRTMAT[,selU[1]],
	dy   = MRTMAT[,selU[2]],
	dz   = MRTMAT[,selU[3]],
	comment=paste("density F",0:18),
	group="f"
)

AddDensity(
	name = paste("T",1:nrow(MRTT)-1,sep=""),
	dx   = MRTT[2,],
	dy   = MRTT[3,],
	dz   = MRTT[4,],
	comment=paste("density T",1:nrow(MRTT)-1),
	group="T"
)

AddDensity(
	name = "w",
	comment="weight fluid-solid",
	group="w",
	parameter=T
)

#AddQuantity( name="Rho",unit="kg/m3")
AddQuantity( name="P",unit="Pa")
AddQuantity( name="U",unit="m/s",vector=T)
AddQuantity( name="T",unit="K")
#AddQuantity( name="RhoB",adjoint=T)
#AddQuantity( name="UB",adjoint=T,vector=T)
#AddQuantity( name="TB",adjoint=T)
AddQuantity( name="W")
AddQuantity( name="WB",adjoint=T)

AddSetting(name="omega", comment='One over relaxation time')
AddSetting(name="nu", omega='1.0/(3*nu + 0.5)', default=1.6666666, comment='Viscosity')
AddSetting(name="InletVelocity", default="0m/s", comment='Inlet velocity')
AddSetting(name="InletPressure", InletDensity='1.0+InletPressure*3', default="0Pa", comment='Inlet pressure')
AddSetting(name="InletDensity", default=1, comment='Inlet density')
AddSetting(name="InletTemperature", comment='Inlet temperature')
AddSetting(name="HeaterTemperature", comment='Temperature of the heater')
AddSetting(name="LimitTemperature", comment='Limit temperature for penalties')
AddSetting(name="FluidAlpha", comment='Heat conductivity of fluid')
AddSetting(name="SolidAlpha", comment='Heat conductivity of solid')
AddSetting(name="HeatSource", comment='Heat input at heat source')
AddSetting(name="Inertia", comment='Inertia of the transport equation')

AddSetting(name="PorocityGamma", comment='Gamma in hiperbolic transformation of porocity (-infty,1)')
AddSetting(name="PorocityTheta", comment='Theta in hiperbolic transformation of porocity', PorocityGamma='1.0 - exp(PorocityTheta)')

AddGlobal(name="HeatFlux", comment='Flux of heat', unit="Km3/s")
AddGlobal(name="HeatSquareFlux", comment='Flux of temperature squered', unit="K2m3/s")
AddGlobal(name="Flux", comment='Volume flux', unit="m3/s")
AddGlobal(name="Temperature", comment='Integral of temperature', unit="K")
AddGlobal(name="HighTemperature", comment='Penalty for high temperature')
AddGlobal(name="LowTemperature", comment='Penalty for low temperature')
AddGlobal(name="MaterialPenalty", comment='Quadratic penalty for intermediate material parameter', unit="m3")
