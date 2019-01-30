####################################
#
#	DISCLAIMER
#
#	This is _very_ simple test environment, made to "look" into model parsing process. 
#	Make project dir R's CWD and source it. Some error may occur, 
#	but you should be able to test R expressions is environment similar to pre-Dynamics.c.Rt
#
#	@mdzik
####################################



printf = function(...) {
  Ucat(sprintf(...))
}
quiet.source = function(..., comment="# ") {
  f=textConnection("quiet.source.text","w");
  sink(f); ret=source(...); sink();
  close(f);
  cat(paste(comment,quiet.source.text,"\n",sep=""),sep="")
  ret
}
if (!exists("include.dir")) include.dir=NULL;
source = function(file,...) {
  pot = c(file, paste(include.dir,file,sep="/"))
  sel = sapply(pot,file.exists)
  sel = which(sel)
  if (length(sel) < 1)
    stop("file not found:",file," in include directories:",paste(include.dir,collapse=","))
  newfile=pot[sel[1]]
  base::source(file=newfile,...)
}
add.include.dir = function(dir) {
  if (substr(dir,1,1) != "/") dir = paste(getwd(),dir,sep="/");
  include.dir <<- c(include.dir,dir)
}
linemark=function(...) {invisible(NULL)}

MODEL='d2q9_pf'

add.include.dir('./tools/')
add.include.dir('./src/')
add.include.dir(paste('./src/',MODEL, sep=''))

source('conf.R')
source('Dynamics.R')
source('lib/feq.R')


source("lib/boundary.R")
source("lib/feq.R")



##########################################################################################
### EXAMPLE: MRT generation

U = t(as.matrix(   rbind( Density$dx[Density$group=='f'], Density$dy[Density$group=='f'] ) ))
EQ = MRT_eq(U, ortogonal=FALSE)
wi = subst(EQ$Req, Jx=0, Jy=0, Jz=0)
wi = subst(wi, rho=1)
wi = gapply(wi,function(x) x$.M, simplify=TRUE)
wi = wi %*% solve(EQ$mat)
wi = as.vector(wi)

W0 = solve(EQ$mat) %*% diag(1/wi) %*% solve(t(EQ$mat))
i = rev(1:nrow(W0))
H = chol(W0[i,i])[i,i]
H = H * c(1,sqrt(3)/3,sqrt(3)/3,sqrt(2),sqrt(2),1,sqrt(6)/3,sqrt(6)/3,2)
B = EQ$mat %*% t(H)

EQ = MRT_eq(U, mat=B)



f = PV(Density$name[Density$group=='f'])
rho = PV("rho")
J = PV("J",c("x","y"))
u = PV(c("u.x","u.y"))


# things related to h
h = PV(Density$name[Density$group=='h'])
pf = PV("pf")
#  phi = PV(paste("phi(",-U[,1],",",-U[,2],")"))
n = PV(c("n.x","n.y"))
c_sq = 1/3.
Bh = PV('Bh')
W = PV("W")	
n = c(PV('n.x'),PV('n.y'))

EQ_h = MRT_eq(U,mat=B)
EQ_h$feq = ( subst( subst(EQ_h$feq, Jx = rho*PV('u.x'), Jy = rho*PV('u.y')), rho=pf ) )

mob = PV("M")
Bh = 3*mob * (1.-4.*pf*pf)*W 

EQ_h$feq = EQ_h$feq +  Bh * wi * n %*% t(U)
EQ_h$Req = EQ_h$feq %*% EQ_h$mat  
UN = t(U[1:9,])
phis = PV(paste('phi(',UN[1,],',',UN[2,],')'))   


k01 = PV("k01")
k10 = PV("k10")

