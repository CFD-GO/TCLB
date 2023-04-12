#################################################
# Preamble to set up variables in R environment #
#################################################
g=PV(DensityAll$name[DensityAll$group=="g"])
wg = PV(paste0("wg[",1:27-1,"]"))
# Extracting velocity set
U = as.matrix(DensityAll[DensityAll$group=="g",c("dx","dy","dz")])
u = PV(c("U","V","W"))
absCi2 = (U[,1]*U[,1]+U[,2]*U[,2]+U[,3]*U[,3])
if (Options$q27){
    PF_velocities = 27
} else {
    PF_velocities = 15
}
# Formulate Weighted-MRT matrix
m0 = matrix(1, 1, 27)
m1 = U[,1]
m2 = U[,2]
m3 = U[,3]
m4 = U[,1]*U[,2]
m5 = U[,2]*U[,3]
m6 = U[,3]*U[,1]
m7 = 3*U[,1]*U[,1] - absCi2
m8 = U[,2]*U[,2] - U[,3]*U[,3]
m9 = absCi2 - 1
m10 = U[,1] * (3*absCi2 - 5)
m11 = U[,2] * (3*absCi2 - 5)
m12 = U[,3] * (3*absCi2 - 5)
m13 = U[,1] * (U[,2]*U[,2] - U[,3]*U[,3])
m14 = U[,2] * (U[,3]*U[,3] - U[,1]*U[,1])
m15 = U[,3] * (U[,1]*U[,1] - U[,2]*U[,2])
m16 = U[,1]*U[,2]*U[,3]
m17 = 0.5 * (3*absCi2*absCi2 - 7*absCi2+2)
m18 = (3*absCi2 - 4) * (3*U[,1]*U[,1] - absCi2)
m19 = (3*absCi2 - 4) * (U[,2]*U[,2] - U[,3]*U[,3])
m20 = U[,1]*U[,2]*(3*absCi2 - 7)
m21 = U[,2]*U[,3]*(3*absCi2 - 7)
m22 = U[,3]*U[,1]*(3*absCi2 - 7)
m23 = 0.5 * U[,1]*(9*absCi2*absCi2 - 33*absCi2 + 26)
m24 = 0.5 * U[,2]*(9*absCi2*absCi2 - 33*absCi2 + 26)
m25 = 0.5 * U[,3]*(9*absCi2*absCi2 - 33*absCi2 + 26)
m26 = 0.5 * (9*absCi2*absCi2*absCi2 - 36*absCi2*absCi2 + 33*absCi2 - 2)

M = rbind(m0,m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,m13,m14,m15,m16,m17,m18,m19,m20,m21,m22,m23,m24,m25,m26)
m = PV(c(paste("m[",1:27-1,"]",sep="")))
invM = solve(M)

geq = PV(c(paste("geq[",1:27-1,"]",sep="")))
Fi  = PV(c(paste("F_i[",1:27-1,"]",sep="")))
h   = PV(c(paste("h",1:PF_velocities-1,sep="")))
heq  = PV(c(paste("heq[",1:PF_velocities-1,"]",sep="")))
Fphi = PV(c(paste("F_phi[",1:PF_velocities-1,"]",sep="")))
gammah = PV(c(paste("Gamma[",1:PF_velocities-1,"]",sep="")))
if (Options$q27){
    w_h = PV(c(paste("wg[",1:27-1,"]",sep="")))
} else {
    w_h = c(0.75, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0)
}
omega = PV("omega_phi")
phase = PV("PhaseF")
gammag = PV(c(paste("Gamma[",1:27-1,"]",sep="")))

if (Options$OutFlow) {
    g_old = PV(Density$name[Density$group == "gold"])
    h_old = PV(Density$name[Density$group == "hold"])
}

##################################################
# Functions to print c-code for grad and laplace #
##################################################
IsotropicGrad <- function(myStringVec, myStringSca){
  cat(paste0(myStringVec,".y = 16.00 * (",myStringSca, "(0,1,0) - ", myStringSca, "(0,-1,0)) + ", myStringSca, "(1,1,1) + ", myStringSca, "(-1,1,1) - ", myStringSca, "(1,-1,1) - ", myStringSca, "(-1,-1,1) + ", myStringSca, "(1,1,-1)+ ", myStringSca, "(-1,1,-1)- ", myStringSca,  "(1,-1,-1)- ", myStringSca, "(-1,-1,-1) +  4.00 * (", myStringSca, "(1,1,0) + ", myStringSca, "(-1,1,0) - ", myStringSca, "(1,-1,0) - ", myStringSca, "(-1,-1,0) +  ", myStringSca, "(0,1,1) - ", myStringSca, "(0,-1,1) + ", myStringSca, "(0,1,-1) - ", myStringSca, "(0,-1,-1));\n"))
  cat(paste0(myStringVec,".x = 16.00 * (",myStringSca, "(1,0,0) - ", myStringSca, "(-1,0,0)) + ", myStringSca, "(1,1,1) - ", myStringSca, "(-1,1,1) + ", myStringSca, "(1,-1,1) - ", myStringSca, "(-1,-1,1) + ", myStringSca, "(1,1,-1)- ", myStringSca, "(-1,1,-1) + ", myStringSca, "(1,-1,-1) - ", myStringSca, "(-1,-1,-1) +  4.00 * (", myStringSca, "(1,1,0) - ", myStringSca, "(-1,1,0) + ", myStringSca, "(1,-1,0) - ", myStringSca, "(-1,-1,0) + ", myStringSca, "(1,0,1) - ", myStringSca, "(-1,0,1) + ", myStringSca, "(1,0,-1) - ", myStringSca, "(-1,0,-1));\n"))
  cat(paste0(myStringVec,".z = 16.00 * (",myStringSca, "(0,0,1) - ", myStringSca, "(0,0,-1)) + ", myStringSca, "(1,1,1) + ", myStringSca, "(-1,1,1) + ", myStringSca, "(1,-1,1) + ", myStringSca, "(-1,-1,1) - ", myStringSca, "(1,1,-1)- ", myStringSca, "(-1,1,-1)- ", myStringSca,  "(1,-1,-1)- ", myStringSca, "(-1,-1,-1) +  4.00 * (", myStringSca, "(1,0,1) + ", myStringSca, "(-1,0,1) - ", myStringSca, "(1,0,-1) - ", myStringSca, "(-1,0,-1) +  ", myStringSca, "(0,1,1) + ", myStringSca, "(0,-1,1) - ", myStringSca, "(0,1,-1) - ", myStringSca, "(0,-1,-1));\n"))
  cat(paste0(myStringVec,".x /= 72.0;\n"))
  cat(paste0(myStringVec,".y /= 72.0;\n"))
  cat(paste0(myStringVec,".z /= 72.0;\n"))
}
myLaplace <- function(myStringVec, myStringSca){
  cat(paste0(myStringVec, " = 16.0 *((", myStringSca, "(1,0,0)) + (", myStringSca, "(-1,0,0)) + (", myStringSca, "(0,1,0)) + (", myStringSca, "(0,-1,0))+ (", myStringSca, "(0,0,1)) + (", myStringSca, "(0,0,-1)))	+ 1.0 *((", myStringSca, "(1,1,1)) + (", myStringSca, "(-1,1,1)) + (", myStringSca, "(1,-1,1))+ (", myStringSca, "(-1,-1,1)) + (", myStringSca, "(1,1,-1))+ (", myStringSca, "(-1,1,-1)) + (", myStringSca, "(1,-1,-1))+(", myStringSca, "(-1,-1,-1))) + 4.0 *((", myStringSca, "(1,1,0)) + (", myStringSca, "(-1,1,0))+ (", myStringSca, "(1,-1,0))+ (", myStringSca, "(-1,-1,0))+ (", myStringSca, "(1,0,1)) + (", myStringSca, "(-1,0,1))+ (", myStringSca, "(1,0,-1))+ (", myStringSca, "(-1,0,-1))+ (", myStringSca, "(0,1,1)) + (", myStringSca, "(0,-1,1))+ (", myStringSca, "(0,1,-1))+ (", myStringSca, "(0,-1,-1))) - 152.0 * ", myStringSca, "(0,0,0);\n"))
  cat(paste0(myStringVec, "/= 36.0;\n"))
}
