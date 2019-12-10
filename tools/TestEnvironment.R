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

MODEL='d3q27_csf'
BASE='multiphase'
#BASE='flow'

MPATH=paste('./models' ,BASE, MODEL, sep="/")
add.include.dir('./tools/')
add.include.dir('./src/')
add.include.dir(MPATH)


source('conf.R')
#source('Dynamics.R')


##########################################################################################
### EXAMPLE: MRT generation
source("conf.R")
c_header();


# Creating variables for symbolic computations
f = PV(DensityAll$name[DensityAll$group=="f"])
rho =  PV("rho")
J = PV("J",c("x","y","z"))
tmp = PV("tmp")

# Extracting velocity set
U = as.matrix(DensityAll[DensityAll$group=="f",c("dx","dy","dz")])

# Calculating equlibrium density set
source("lib/feq.R")
source("lib/boundary.R")

EQ = MRT_eq(U, rho, J, ortogonal=FALSE);
#	EQ = MRT_eq(U, rho, J);






