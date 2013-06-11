if (!exists("ADJOINT")) ADJOINT=0
if (!exists("DOUBLE")) DOUBLE=0

source("tools/fun_v3.R")

table_from_text = function(text) {
	con = textConnection(text);
	tab = read.table(con, header=T);
	close(con);
	tab
}

Settings = data.frame(
        name = c("omega","nu","UX_mid"),
        derived = c(NA,"omega",NA),
        equation = c(NA,"1.0/(3*nu + 0.5)",NA),
        comment = c("one over relaxation time", "viscosity", "velocity on inlet")
)

Globals = table_from_text("
        name            in_objective   comment
        PressureLoss    1              'pressure loss'
")

Node_Group = c(
  NONE        =0x0000
, COLLISION   =0x0070
, BOUNDARY    =0x000F
, ADDITIONALS =0x0F00
, OPTIMIZATION=0xF000
, OBJECTIVE   =0x7000
, DESIGNSPACE =0x8000
, ALL         =0xFFFF
)

Node = c(
  None        =0x0000
, BGK         =0x0010
, MRT         =0x0020
, MR          =0x0030
, Entropic    =0x0040
, Default     =0x0020

, Wall        =0x0001
, Solid       =0x0002
, WVelocity   =0x0003
, WPressure   =0x0004
, WPressureL  =0x0005
, EPressure   =0x0006
, EVelocity   =0x0007
, MovingWall  =0x0008

, Heater      =0x0100
, HeatSource  =0x0200
, Wet         =0x0300
, Dry         =0x0400

, Inlet       =0x1000
, Outlet      =0x2000
, Obj1        =0x3000
, Obj2        =0x4000
, Obj3        =0x5000
, Thermometer =0x6000

, DesignSpace =0x8000
)

source("Dynamics.R")

if (! "unit" %in% names(Quantities)) {
	Quantities$unit = "1"
} else {
	Quantities$unit = as.character(Quantities$unit)
}

if (! "adjoint" %in% names(Quantities)) {
	Quantities$adjoint = F
}

ifdef.global.mark = F
ifdef = function(val=F, tag="ADJOINT") {
	if ((!ifdef.global.mark) && ( val)) cat("#ifdef",tag,"\n");
	if (( ifdef.global.mark) && (!val)) cat("#endif //",tag,"\n");
	ifdef.global.mark <<- val
}

if (ADJOINT==1) {
	Density$adjoint = F
	DensityAD = Density
	DensityAD$dx = -Density$dx
	DensityAD$dy = -Density$dy
	DensityAD$dz = -Density$dz
	DensityAD$name = as.character(DensityAD$name);
	i = grepl("[[]", Density$name)
	DensityAD$name[i] = sub("[[]","b[", Density$name[i])
	DensityAD$name[!i] = paste(Density$name[!i], "b",sep="")
	DensityAD$adjoint = T
	DensityAll = rbind(Density,DensityAD)

	Settings = rbind(Settings, data.frame(
		name=paste(Globals$name,"InObj",sep=""),
		derived=NA,equation=NA,comment=Globals$comment))
} else {
	DensityAD = NULL
	DensityAll = Density
}

DensityAll$nicename = gsub("[][ ]","",DensityAll$name)


Margin = data.frame(
	name = paste("margin",1:27,sep=""),
	side = paste("side",1:27,sep=""),
	dx   = rep(-1:1,times=9),
	dy   = rep(rep(-1:1,times=3),each=3),
	dz   = rep(-1:1,each=9),
	command=paste("Margin",1:27)
)

# Margin = Margin[1:19,]

GetMargins = function(dx,dy,dz) {
	fun = function(dx,dy,dz) {
		w = c(dx,dy,dz)
		if (any(w[!is.na(w)] == 0)) {
			0
		} else {
			w[is.na(w)] = 0
			(w[1]+1) + (w[2]+1)*3 + (w[3]+1)*9 + 1
		}
	}
	c(
		fun(dx,NA,NA),
		fun(NA,dy,NA),
		fun(NA,NA,dz),
		fun(dx,dy,NA),
		fun(NA,dy,dz),
		fun(dx,NA,dz),
		fun(dx,dy,dz)
	)
}

Margin$size = 0

nx = PV("nx");
ny = PV("ny");
nz = PV("nz");
SideSize = rbind(ny*nz, nx*nz, nx*ny, nz, nx, ny, 1);
zero = PV(0);
MarginSize =  zero
for (i in 2:nrow(Margin)) MarginSize = rbind(MarginSize, zero)

x = PV("node.x");
y = PV("node.y");
z = PV("node.z");
SideOffset = rbind(y + z*ny, x + z*nx, x + y*nx, z, x, y, 0);
MarginOffset =  zero
for (i in 2:nrow(Margin)) MarginOffset = rbind(MarginOffset, zero)


for (i in 1:nrow(DensityAll))
{
	x = DensityAll[i,]
	w = GetMargins(x$dx,x$dy,x$dz)
	for (k in 1:length(w)) {j = w[k];
		if (j != 0) {
			Margin$size[j] = Margin$size[j] + 1
			MarginSize[[j]] = MarginSize[[j]] + SideSize[[k]]
			MarginOffset[[j]] = SideOffset[[k]]
		}
	}
}



NonEmptyMargin = which(Margin$size != 0)

Settings$FunName = paste("SetConst",Settings$name,sep="_")

git_version = function(){f=pipe("git describe --always --tags"); v=readLines(f); close(f); v}
version=git_version()

clb_header = c(
sprintf("-------------------------------------------------------------"),
sprintf("   CLB                                                       "),
sprintf("    CUDA Lattice Boltzmann                                   "),
sprintf("    Author: Lukasz Laniewski-Wollk                           "),
sprintf("    Developed at: Warsaw University of Technology - 2012     "),
sprintf("-------------------------------------------------------------")
)


c_header = function() {
	for (l in clb_header)
	cat("/*",l,"*/\n",sep="");
}

hash_header = function() {
	for (l in clb_header)
	cat("#",l,"\n",sep="");
}