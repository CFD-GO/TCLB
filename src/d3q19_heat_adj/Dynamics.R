
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

Density = data.frame(
	name = paste("f",0:18,sep=""),
	dx   = MRTMAT[,selU[1]],
	dy   = MRTMAT[,selU[2]],
	dz   = MRTMAT[,selU[3]],
	command=paste("density F",0:18),
	group="f"
)

f = PV(Density$name)

Density = rbind(Density,data.frame(
	name = paste("T",1:nrow(MRTT)-1,sep=""),
	dx   = MRTT[2,],
	dy   = MRTT[3,],
	dz   = MRTT[4,],
	command=paste("density T",1:nrow(MRTT)-1),
	group="T"
))

Density = rbind(Density,data.frame(
	name = "w",
	dx   = 0,
	dy   = 0,
	dz   = 0,
	command="weight fluid-solid",
	group="w"
))

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


Quantities = data.frame(
        name = c("Rho","U","T","RhoB","UB","TB","W","WB"),
        type = c("real_t","vector_t","real_t","real_t","vector_t","real_t","real_t","real_t"),
	adjoint = c(F,F,F,T,T,T,F,T)
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
        Flux    1              'pressure loss'
	Temperature 1 'integral of temperature'
	HighTemperature 1 'penalty for high temperature'
	LowTemperature  1 'penalty for low temperature'
")
