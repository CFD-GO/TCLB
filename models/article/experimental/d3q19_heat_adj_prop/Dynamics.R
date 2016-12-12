
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

f = PV(Density$name)

AddDensity(
	name = paste("T",1:nrow(MRTT)-1,sep=""),
	dx   = MRTT[2,],
	dy   = MRTT[3,],
	dz   = MRTT[4,],
	comment=paste("density T",1:nrow(MRTT)-1),
	group="T"
)

AddDensity(
	name = paste("w",1:2-1,sep=""),
	dx=c(-1,1),
	comment="weight fluid-solid moving in X",
	group="wm"
)

AddDensity(
	name = "w",
	comment="weight fluid-solid",
	group="w",
	parameter=T
)

f = PV(Density$name[Density$group=="f"])
fT = PV(Density$name[Density$group=="T"])


rho = PV("rho")
J = PV(c("Jx","Jy","Jz"))
rho0 = 1

if (FALSE) {
	we = 0
	weJ = -475/63
	wxx = 0
} else {
	we = 3
	weJ = -11/2
	wxx = -1/2
}

pxx = 1/(3*rho0) * (J[1]*J[1]*2 - J[2] * J[2] - J[3] * J[3]) 
pww = 1/(rho0) * (J[2] * J[2] - J[3] * J[3]) 
pxy = 1/(rho0) * (J[1]*J[2]) 
pyz = 1/(rho0) * (J[2]*J[3]) 
pxz = 1/(rho0) * (J[1]*J[3]) 

Req = rbind(
	rho,
	-11*rho + 19/rho0*sum(J*J),
	we*rho + weJ/rho0*sum(J*J),
	J[1],
	-2/3*J[1],
	J[2],
	-2/3*J[2],
	J[3],
	-2/3*J[3],
	pxx*3,
	wxx*pxx*3,
	pww,
	wxx*pww,
	pxy,
	pyz,
	pxz,
	0,
	0,
	0
)

U = MRTMAT[,selU]
#f = PV(Density$name)
R = PV(paste("R",0:18,sep=""))


R[1] = rho
R[c(4,6,8)] = J
R[-c(1,4,6,8)] = PV(paste("R",0:14,sep=""))
selR = c(2,3,5,7,9:19)

#R[[1]] = rho[[1]]
#R[[4]] = J[[1]]
#R[[6]] = J[[2]]
#R[[8]] = J[[3]]


renum = c(19, 1, 2, 3, 4, 5, 6, 7, 11, 8, 12, 9, 13, 10, 14, 15, 17, 16, 18)

I = rep(0, 19)
I[renum] = 1:19

if (FALSE) {
Sy = rbind(
	PV(0),
	PV(1.19),
	PV(1.4),
	PV(0),
	PV(1.2),
	PV(0),
	PV(1.2),
	PV(0),
	PV(1.2),
	PV("omega"),
	PV(1.4),
	PV("omega"),
	PV(1.4),
	PV("omega"),
	PV("omega"),
	PV("omega"),
	PV(1.98),
	PV(1.98),
	PV(1.98)
)
	
}


AddQuantity( name="Rho",unit="kg/m3")
AddQuantity( name="U",unit="m/s",vector=T)
AddQuantity( name="T",unit="K")
#AddQuantity( name="RhoB",adjoint=T)
#AddQuantity( name="UB",adjoint=T,vector=T)
#AddQuantity( name="TB",adjoint=T)
AddQuantity( name="W")
AddQuantity( name="W0")
AddQuantity( name="WB",adjoint=T)

AddSetting(name="omega", comment='one over relaxation time')
AddSetting(name="nu", omega='1.0/(3*nu + 0.5)', default=0.16666666, comment='viscosity')
AddSetting(name="InletVelocity", default="0m/s", comment='inlet velocity')
AddSetting(name="InletPressure", InletDensity='1.0+InletPressure/3', default="0Pa", comment='inlet pressure')
AddSetting(name="InletDensity", default=1, comment='inlet density')
AddSetting(name="InletTemperature", comment='inlet temperature')
AddSetting(name="HeaterTemperature", comment='temperature of the heater')
AddSetting(name="LimitTemperature", comment='temperature of the heater')
AddSetting(name="FluidAlpha", comment='heat conductivity of fluid')
AddSetting(name="SolidAlpha", comment='heat conductivity of fluid')
AddSetting(name="HeatSource", comment='heat conductivity of fluid')
AddSetting(name="Inertia", comment='inertia of the transport equation')

AddGlobal(name="HeatFlux", comment='pressure loss')
AddGlobal(name="HeatSquareFlux", comment='pressure loss')
AddGlobal(name="Flux", comment='pressure loss')
AddGlobal(name="Temperature", comment='integral of temperature')
AddGlobal(name="HighTemperature", comment='penalty for high temperature')
AddGlobal(name="LowTemperature", comment='penalty for low temperature')
AddGlobal(name="MaterialPenalty", comment='quadratic penalty for intermediate material parameter')

AddSetting(name="PropagateX", comment='inertia of the transport equation')
