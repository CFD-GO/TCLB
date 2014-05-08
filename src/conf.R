#-----------------------------------------
#  Main R file.
#    defines all additiona functions
#    reads model-specific configuration
#    creates all needed tables of:
#      Densities
#      Quantities
#      Settings
#      Globals
#      Consts
#-----------------------------------------

if (!exists("ADJOINT")) ADJOINT=0
if (!exists("DOUBLE")) DOUBLE=0

source("fun_v3.R")
source("bunch.R")

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

c_table_decl = function(d) {
	d = as.character(d)
	sel = grepl("\\[",d)
	if(any(sel)) {
		w = d[sel]
#		w = regmatches(w,regexec("([^[]*)\\[([^\\]]*)]",w))
		r = regexpr("\\[([^\\]]*)]",w)
		w = lapply(1:length(r), function(i) {
			a_=w[i]
			c(a_,
				substr(a_,1,r[i]-1),
				substr(a_,r[i]+1,r[i]+attr(r,"match.length")[i]-2)
			)
		})

		w = do.call(rbind,w)
		w = data.frame(w)
		w[,3] = as.integer(as.character(w[,3]))
		w = by(w,w[,2],function(x) {paste(x[1,2],"[",max(x[,3])+1,"]",sep="")})
		w = do.call(c,as.list(w))
	} else {
		w = c()
	}
	w = c(w,d[!sel])
	w
}


ifdef.global.mark = F
ifdef = function(val=F, tag="ADJOINT") {
	if ((!ifdef.global.mark) && ( val)) cat("#ifdef",tag,"\n");
	if (( ifdef.global.mark) && (!val)) cat("#endif //",tag,"\n");
	ifdef.global.mark <<- val
}


DensityAll = data.frame()
Globals = data.frame()
Settings = data.frame()
Quantities = data.frame()
NodeTypes = data.frame()
Fields = data.frame()


AddDensity = function(name, dx=0, dy=0, dz=0, comment="", field=name, adjoint=F, group="", parameter=F) {
	if (any((parameter) && (dx != 0) && (dy != 0) && (dz != 0))) stop("Parameters cannot be streamed (AddDensity)");
	if (missing(name)) stop("Have to supply name in AddDensity!")
	comment = ifelse(comment == "", name, comment);
	d = data.frame(
		name=name,
		field=field,
		dx=dx,
		dy=dy,
		dz=dz,
		comment=comment,
		adjoint=adjoint,
		group=group,
		parameter=parameter
	)
	DensityAll <<- rbind(DensityAll,d)
	if (any(Fields$name == field)) {
		i = which(Fields$name == field)
		Fields$minx[i] <<- min(Fields$minx[i], dx)
		Fields$maxx[i] <<- max(Fields$maxx[i], dx)
		Fields$miny[i] <<- min(Fields$miny[i], dy)
		Fields$maxy[i] <<- max(Fields$maxy[i], dy)
		Fields$minz[i] <<- min(Fields$minz[i], dz)
		Fields$maxz[i] <<- max(Fields$maxz[i], dz)
	} else {
		d = data.frame(
			name=field,
			minx=dx,maxx=dx,
			miny=dy,maxy=dy,
			minz=dz,maxz=dz,
			comment=comment,
			adjoint=adjoint,
			group=group,
			parameter=parameter
		)
		Fields <<- rbind(Fields, d)
	}
}

AddField = function(name, stencil2d=0, stencil3d=0, dx=0, dy=0, dz=0, comment="", adjoint=F, group="", parameter=F) {
	if (missing(name)) stop("Have to supply name in AddField!")
	comment = ifelse(comment == "", name, comment);

		d = data.frame(
			name=name,
			minx=min(dx,-stencil2d,-stencil3d),
			maxx=max(dx, stencil2d, stencil3d),
			miny=min(dy,-stencil2d,-stencil3d),
			maxy=max(dy, stencil2d, stencil3d),
			minz=min(dz,           -stencil3d),
			maxz=max(dz,            stencil3d),
			comment=comment,
			adjoint=adjoint,
			group=group,
			parameter=parameter
		)
		Fields <<- rbind(Fields, d)
}


AddSetting = function(name,  comment, default=0, unit="1", adjoint=F, derived, equation, ...) {
	if (missing(name)) stop("Have to supply name in AddSetting!")
	if (missing(comment)) {
		comment = name
	}
	if (missing(derived)) {
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
	} else {
		if (missing(equation)) stop("'derived' provided, but no 'equation' in AddSetting")
	}
	s = data.frame(
		name=name,
		derived=derived,
		equation=equation,
		unit=unit,
		default=default,
		adjoint=adjoint,
		comment=comment
	)
	Settings <<- rbind(Settings,s)
}

AddGlobal = function(name, var, comment="", unit="1", adjoint=F) {
	if (missing(name)) stop("Have to supply name in AddGlobal!")
	if (missing(var)) var=name
	if (comment == "") {
		comment = name
	}
	g = data.frame(
		name=name,
		var=var,
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

AddNodeType = function(name, group) {
	NodeTypes <<- rbind(NodeTypes, data.frame(
		name=name,
		group=group
	))
}

AddNodeType("BGK","COLLISION")
AddNodeType("MRT","COLLISION")
AddNodeType("MR","COLLISION")
AddNodeType("Entropic","COLLISION")
AddNodeType("Wall","BOUNDARY")
AddNodeType("Solid","BOUNDARY")
AddNodeType("WVelocity","BOUNDARY")
AddNodeType("WPressure","BOUNDARY")
AddNodeType("WPressureL","BOUNDARY")
AddNodeType("EPressure","BOUNDARY")
AddNodeType("EVelocity","BOUNDARY")
AddNodeType("MovingWall","BOUNDARY")
AddNodeType("Heater","ADDITIONALS")
AddNodeType("HeatSource","ADDITIONALS")
AddNodeType("Wet","ADDITIONALS")
AddNodeType("Dry","ADDITIONALS")
AddNodeType("Propagate","ADDITIONALS")
AddNodeType("Inlet","OBJECTIVE")
AddNodeType("Outlet","OBJECTIVE")
AddNodeType("Obj1","OBJECTIVE")
AddNodeType("Obj2","OBJECTIVE")
AddNodeType("Obj3","OBJECTIVE")
AddNodeType("Thermometer","OBJECTIVE")
AddNodeType("DesignSpace","DESIGNSPACE")

source("Dynamics.R") #------------------------------------------- HERE ARE THE MODEL THINGS

NodeShift = 1
NodeShiftNum = 0
NodeTypes = unique(NodeTypes)
NodeTypes = do.call(rbind, by(NodeTypes,NodeTypes$group,function(tab) {
	n = nrow(tab)
	l = ceiling(log2(n+1))
	tab$index = 1:n
	tab$Index = tab$name
	tab$value = NodeShift*(1:n)
	tab$mask  = NodeShift*((2^l)-1)
	tab$shift = NodeShiftNum
	NodeShift    <<- NodeShift * (2^l)
	NodeShiftNum <<- NodeShiftNum + l
	tab
}))

if (any(NodeTypes$value >= 2^16)) stop("NodeTypes exceeds short int")

Node=NodeTypes$value
names(Node) = NodeTypes$name
Node["None"] = 0

i = !duplicated(NodeTypes$group)
Node_Group=NodeTypes$mask[i]
names(Node_Group) = NodeTypes$group[i]
Node_Group["ALL"] = sum(Node_Group)


Scales = data.frame(name=c("dx","dt","dm"), unit=c("m","s","kg"));

if (ADJOINT==1) {
	for (d in rows(DensityAll)) {
		n = as.character(d$name)
		if (grepl("[[]", n)) {
			n = sub("[[]","b[", n)
		} else {
			n = paste(n, "b", sep="")
		}
		AddDensity(
			name=n,
			dx=-d$dx,
			dy=-d$dy,
			dz=-d$dz,
			comment=paste("adjoint to",d$comment),
			group=d$group,
			parameter=d$parameter,
			adjoint=T
		)
	}
	for (s in rows(Settings)) {
		AddGlobal(
			name=paste(s$name,"_D",sep=""),
			var=paste(s$name,"b",sep=""),
			comment=paste("Gradient of objective with respect to [",s$comment,"]",sep=""),
			adjoint=T
		)
	}
	for (g in rows(Globals)) if (! g$adjoint){
		AddSetting(
			name=paste(g$name,"InObj",sep=""),
			comment=paste("Weight of [",g$comment,"] in objective",sep=""),
			adjoint=T
		)
	}
	AddSetting(name="Descent",        comment="Optimization Descent", adjoint=T)
	AddSetting(name="GradientSmooth", comment="Gradient smoothing in OptSolve", adjoint=T)
	AddGlobal(name="AdjointRes", comment="square L2 norm of adjoint change", adjoint=T)
}

DensityAll$nicename = gsub("[][ ]","",DensityAll$name)
Density   = DensityAll[! DensityAll$adjoint, ]
DensityAD = DensityAll[  DensityAll$adjoint, ]

Fields$nicename = gsub("[][ ]","",Fields$name)

Fields = bunch(Fields)

AddSetting(name="Threshold", comment="Parameters threshold", default=0.5)

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

Margin$side = c(
	 8, 7, 8,
	 6, 5, 6,
	 8, 7, 8,
	 4, 3, 4,
	 2, 1, 2,
	 4, 3, 4,
	 8, 7, 8,
	 6, 5, 6,
	 8, 7, 8
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
zero = PV(0);
x = PV("node.x");
y = PV("node.y");
z = PV("node.z");

SideSize = rbind(nx*ny*nz, ny*nz, nx*nz, nz, nx*ny, ny, nx, 1);
SideOffset = rbind(x + y*nx + z*nx*ny, y + z*ny, x + z*nx, z, x + y*nx, y, x, 0);

for (i in 1:length(Margin)) {
	Margin[[i]]$Size = zero
	Margin[[i]]$Offset = zero
	Margin[[i]]$opposite_side = Margin[[28-i]]$side
}

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

Dispatch = data.frame(
	Globals=c(   "No",    "No",  "Globs",  "Obj",   "No",      "Globs",    "No",       "Globs",   "No",      "Globs"),
	Action =c(   "No",  "Init",     "No",   "No",  "Adj",        "Adj",   "Adj",         "Adj",  "Opt",        "Opt"),
	Stream =c(   "No",    "No",     "No",   "No",  "Adj",        "Adj",   "Adj",         "Adj",  "Opt",        "Opt"),
	globals=c(  FALSE,   FALSE,     TRUE,   TRUE,  FALSE,         TRUE,   FALSE,          TRUE,  FALSE,         TRUE),
	adjoint=c(  FALSE,   FALSE,    FALSE,  FALSE,   TRUE,         TRUE,    TRUE,          TRUE,   TRUE,         TRUE),
	zeropar=c(  FALSE,   FALSE,    FALSE,  FALSE,  FALSE,        FALSE,    TRUE,          TRUE,   TRUE,         TRUE),
	suffix =c(     "", "_Init", "_Globs", "_Obj", "_Adj", "_Globs_Adj", "_SAdj", "_Globs_SAdj", "_Opt", "_Globs_Opt")
)

Dispatch$adjoint_ver = Dispatch$adjoint
Dispatch$adjoint_ver[Dispatch$Globals == "Obj"] = TRUE

Consts = NULL
for (n in c("Settings","DensityAll","Density","DensityAD","Globals","Quantities","Scales","Fields")) {
	v = get(n)
	if (is.null(v)) v = data.frame()
	Consts = rbind(Consts, data.frame(name=toupper(n), value=nrow(v)));
	if (nrow(v) > 0) {
		v$index = 1:nrow(v)-1
		v$nicename = gsub("[][ ]","",v$name)
		v$Index = paste(" ",toupper(n), "_", v$nicename, " ", sep="")
		Consts = rbind(Consts, data.frame(name=v$Index, value=v$index));
		assign(n,v)
	}
	assign(n,v)
}

GlobalsD = Globals[-nrow(Globals),]


for (i in 1:length(Margin)) {
	Margin[[i]]$Size = zero
	Margin[[i]]$Offset = zero
	Margin[[i]]$opposite_side = Margin[[28-i]]$side
}

offset.fun = function(j_) {
	baseOffset = do.call(rbind,lapply(Margin, function(x) x$Size))
	nx = PV("nx");
	ny = PV("ny");
	nz = PV("nz");
	i = sapply(Margin, function(x) x$side)
	if (missing(j_)) {
		j = NULL
	} else {
		j= (1:length(baseOffset))[-j_]
	}
	function(x,y,z) {
		elementsOffset = rbind(x + y*nx + z*nx*ny, y + z*ny, x + z*nx, z, x + y*nx, y, x, 0);
		ret = elementsOffset[i] + baseOffset
		ret[j] = PV(0)
		ret
	}
}

Fields$Offset = rep(list(NULL),length(Fields))

save.image(file="test.Rdata")

for (x in Fields)
{
	w = c(	GetMargins(x$minx,x$miny,x$minz),
		GetMargins(x$maxx,x$maxy,x$maxz) )
	w = unique(w[w !=0])
	Fields$Offset[[x$index+1]] = offset.fun(w)
	for (j in w) {
		Margin[[j]]$size   = Margin[[j]]$size + 1
		Margin[[j]]$Size   = Margin[[j]]$Size + SideSize[Margin[[j]]$side]
		Margin[[j]]$Offset = SideOffset[Margin[[j]]$side]
	}
}






git_version = function(){f=pipe("git describe --always --tags"); v=readLines(f); close(f); v}

clb_header = c(
sprintf("-------------------------------------------------------------"),
sprintf("  CLB - Cudne LB - Stencil Version                           "),
sprintf("     CUDA based Adjoint Lattice Boltzmann Solver             "),
sprintf("     Author: Lukasz Laniewski-Wollk                          "),
sprintf("     Developed at: Warsaw University of Technology - 2012    "),
sprintf("-------------------------------------------------------------")
)

c_header = function() {
#	for (l in clb_header)
	cat(paste("/*",clb_header,"*/",collapse="\n",sep=""),sep="");
	cat("\n");
}

hash_header = function() {
	for (l in clb_header)
	cat("# |",l,"|\n",sep="");
	cat("\n");
}
