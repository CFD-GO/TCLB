if (!exists("ADJOINT")) ADJOINT=0
if (!exists("DOUBLE")) DOUBLE=0

source("tools/fun_v3.R")

rows = function(x) {
	rows_df= function(x) {
		if (nrow(x) > 0) {
			lapply(1:nrow(x),function(i) unclass(x[i,,drop=F]))
		} else {
			list()
		}
	};
	switch( class(x),
		list       = x,
		data.frame = rows_df(x)
	)
}

table_from_text = function(text) {
	con = textConnection(text);
	tab = read.table(con, header=T);
	close(con);
	tab
}

Density = NULL
Globals = NULL
Settings = NULL
Quantities = NULL


AddDensity = function(name, dx=0, dy=0, dz=0, comment="", adjoint=F, group="", parameter=F) {
	if (parameter) {
		if ((dx != 0) && (dy != 0) && (dz != 0)) stop("Parameters cannot be streamed (AddDensity)");
	}
	if (missing(name)) stop("Have to supply name in AddDensity!")
	if (comment == "") {
		comment = name
	}
	d = data.frame(
		name=name,
		dx=dx,
		dy=dy,
		dz=dz,
		comment=comment,
		adjoint=adjoint,
		group=group,
		parameter=parameter
	)
	Density <<- rbind(Density,d)
}

AddSetting = function(name,  comment="", default=0, ...) {
	if (missing(name)) stop("Have to supply name in AddSetting!")
	if (comment == "") {
		comment = name
	}
	der = list(...)
	if (length(der) == 0) {
		derived = NA;
		equation = NA;
	} else if (length(der) == 1) {
		derived = names(der);
		equation = as.character(der[[1]]);
	} else {
		stop("Only one derived setting allowed in AddSetting!");
	} 
	s = data.frame(
		name=name,
		derived=derived,
		equation=equation,
		default=default,
		comment=comment
	)
	Settings <<- rbind(Settings,s)
}

AddGlobal = function(name, comment="", unit="1", adjoint=F) {
	if (missing(name)) stop("Have to supply name in AddGlobal!")
	if (comment == "") {
		comment = name
	}
	g = data.frame(
		name=name,
		comment=comment,
		unit=unit,
		adjoint=adjoint
	)
	Globals <<- rbind(Globals,g)
}

AddQuantity = function(name, unit="1", vector=F, comment="", adjoint=F) {
	if (missing(name)) stop("Have to supply name in AddQuantity!")
	if (comment == "") {
		comment = name
	}
	if (vector) {
		type="vector_t"
	} else {
		type="real_t"
	}
	q = data.frame(
		name=name,
		type=type,
		unit=unit,
		adjoint=adjoint,
		comment=comment
	)
	Quantities <<- rbind(Quantities,q)
}	

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
if (! "unit" %in% names(Globals)) {
	Globals$unit = "1"
} else {
	Globals$unit = as.character(Globals$unit)
}
if (! "adjoint" %in% names(Globals)) {
	Globals$adjoint = FALSE
} 
if (! "default" %in% names(Settings)) {
	Settings$default = "0"
} else {
	Settings$default = as.character(Settings$default)
}
if (! "unit" %in% names(Settings)) {
	Settings$unit = "1"
} else {
	Settings$unit = as.character(Settings$unit)
}

Scales = data.frame(name=c("dx","dt","dm"), unit=c("m","s","kg"));


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
		derived=NA,equation=NA,comment=Globals$comment,default="0", unit="1"))
} else {
	DensityAD = NULL
	DensityAll = Density
}

DensityAll$nicename = gsub("[][ ]","",DensityAll$name)

GlobalsD = Globals
AddGlobal(name="Objective",comment="Objective function");

Margin = data.frame(
	name = paste("block",1:27,sep=""),
	side = paste("side",1:27,sep=""),
	dx   = rep(-1:1,times=9),
	dy   = rep(rep(-1:1,times=3),each=3),
	dz   = rep(-1:1,each=9),
	command=paste("Margin",1:27)
)

Margin$size = 0
Margin=rows(Margin)


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
		fun(NA,NA,NA),
		fun(dx,NA,NA),
		fun(NA,dy,NA),
		fun(dx,dy,NA),
		fun(NA,NA,dz),
		fun(dx,NA,dz),
		fun(NA,dy,dz),
		fun(dx,dy,dz)
	)
}

nx = PV("nx");
ny = PV("ny");
nz = PV("nz");
SideSize = rbind(nx*ny*nz, ny*nz, nx*nz, nz, nx*ny, ny, nx, 1);
zero = PV(0);

for (i in 1:length(Margin)) {
	Margin[[i]]$Size = zero
	Margin[[i]]$Offset = zero
	Margin[[i]]$opposite_side = Margin[[28-i]]$side
}

x = PV("node.x");
y = PV("node.y");
z = PV("node.z");
SideOffset = rbind(x + y*nx + z*nx*ny, y + z*ny, x + z*nx, z, x + y*nx, y, x, 0);

for (x in rows(Density))
{
	w = GetMargins(x$dx,x$dy,x$dz)
	for (k in 1:length(w)) {
		j = w[k];
		if (j != 0) {
			Margin[[j]]$size   = Margin[[j]]$size + 1
			Margin[[j]]$Size   = Margin[[j]]$Size + SideSize[k]
			Margin[[j]]$Offset = SideOffset[k]
		}
	}
}


NonEmptyMargin = sapply(Margin, function(m) m$size != 0)
NonEmptyMargin = Margin[NonEmptyMargin]



Settings$FunName = paste("SetConst",Settings$name,sep="_")

Dispatch = expand.grid(globals=c(FALSE,TRUE), adjoint=c(FALSE,TRUE))
Dispatch$suffix = paste(
	ifelse(Dispatch$globals,"_Globs",""),
	ifelse(Dispatch$adjoint,"_Adj",""),
	sep=""
)

Settings$index = 1:nrow(Settings)-1
Density$index = 1:nrow(Density)-1
Globals$index = 1:nrow(Globals)-1
Quantities$index = 1:nrow(Quantities)-1


git_version = function(){f=pipe("git describe --always --tags"); v=readLines(f); close(f); v}
version=git_version()

clb_header = c(
sprintf("-------------------------------------------------------------"),
sprintf("  CLB - Cudne LB                                             "),
sprintf("     CUDA based Adjoint Lattice Boltzmann Solver             "),
sprintf("     Author: Lukasz Laniewski-Wollk                          "),
sprintf("     Developed at: Warsaw University of Technology - 2012    "),
sprintf("-------------------------------------------------------------")
)


c_header = function() {
	for (l in clb_header)
	cat("/*",l,"*/\n",sep="");
}

hash_header = function() {
	for (l in clb_header)
	cat("# |",l,"|\n",sep="");
}
