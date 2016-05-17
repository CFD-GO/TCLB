

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

f = PV(Density$name)

AddDensity(
	name = "w",
	comment="weight fluid-solid",
	group="w",
	parameter=T
)

f = PV(Density$name[Density$group=="f"])

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
AddQuantity( name="W")
AddQuantity( name="WB",adjoint=T)

AddSetting(name="omega", comment='one over relaxation time')
AddSetting(name="nu", omega='1.0/(3*nu + 0.5)', default=0.16666666, comment='viscosity')
AddSetting(name="InletVelocity", default="0m/s", comment='inlet velocity')
AddSetting(name="InletPressure", InletDensity='1.0+InletPressure/3', default="0Pa", comment='inlet pressure')
AddSetting(name="InletDensity", default=1, comment='inlet density')
AddSetting(name="Theta", default=1, comment='inlet density')

AddGlobal(name="Flux", comment='pressure loss')
AddGlobal(name="EnergyFlux", comment='pressure loss')
AddGlobal(name="PressureFlux", comment='pressure loss')
AddGlobal(name="PressureDiff", comment='pressure loss')
AddGlobal(name="MaterialPenalty", comment='quadratic penalty for intermediate material parameter')
