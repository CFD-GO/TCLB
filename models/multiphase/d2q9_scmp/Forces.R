library(polyAlgebra);
setwd("TCLB");
source("./CLB/d2q9_scmp_LycettLuo_ViscositySmooth_WMRT/Dynamics.c-debug.R")
Force = PV(c("F.x","F.y"));



tmp = PV(paste("tmp[",1:9-1,"]",sep=""))

EQ_HO = MRT_eq(U, rho, J, ortogonal=FALSE, order=12, mat=M);
EQ_LO = MRT_eq(U, rho, J, ortogonal=FALSE, mat=M);

EQ_dF_HO = MRT_eq(U, rho, J+Force, ortogonal=FALSE, order=12, mat=M);
EQ_dF_LO = MRT_eq(U, rho, J+Force, ortogonal=FALSE, mat=M);


dF_EDM_HO = EQ_dF_HO$Req - EQ_HO$Req
dF_EDM_LO = EQ_dF_LO$Req - EQ_LO$Req

C(tmp,dF_EDM_HO)
C(tmp,dF_EDM_LO)


dF_EDM = EQ_dF$Req - EQ$Req
C(tmp, dF_EDM %*% solve(EQ$mat))
C(tmp,dF_EDM)

df_LL = PV(paste("tmp[",1:9-1,"]",sep=""))
gamma = PV('gamma')
cs2 = 1/3
Fx = Force[1]
Fy = Force[2]
tau = PV('tau')
G = PV("G")
kappa = PV("kappa")

pow <- function(s,c) {
  temp_C = s
  for (ii in 1:(c-1)){ temp_C = temp_C * s}
  return(temp_C)
}

Phi_xx = PV("Phi_xx")
Phi_xy = PV("Phi_xy")
Phi_yx = PV("Phi_yx")
Phi_yy = PV("Phi_yy")



for (i in 1:9){
    vx = U[i,1]
    vy = U[i,2]
    Ux = J[1]*rho^-1
    Uy = J[2]*rho^-1
    wi = wi_[i]

    df_LL[i] = (3.0/2.0)*wi*(0-gamma*tau*(pow(Fx, 2) + pow(Fy, 2) - 3*pow(Fx*vx + Fy*vy, 2)) - 2*rho*tau*(Fx*(Ux - vx) + Fy*(Uy - vy) - 3*(Fx*vx + Fy*vy)*(Ux*vx + Uy*vy)) + rho*(Phi_xx*(3*pow(vx, 2) - 1) + 3*Phi_xy*vx*vy + 3*Phi_yx*vx*vy + Phi_yy*(3*pow(vy, 2) - 1)))*(rho*tau)^-1
}

C(tmp, df_LL)


C(tmp,dF_EDM-df_LL%*% EQ$mat)
C(tmp,dF)


phi = ph

i = 1
j = 1

t1 = (phi-phi[[1]])*U[,i]*U[,j]
t2 = 2*(phi-phi[1])
C( tmp[[1]], G*cs2*phi[1]*(0-kappa*0.5*(wi_%*%t1)) + G*cs2*phi[1]*(kappa+1)*(1*(i==j))*(wi_%*%t2)*(12^-1) )

