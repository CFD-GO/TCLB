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
if (!exists("SYMALGEBRA")) SYMALGEBRA=FALSE

# SYMALGEBRA=TRUE

options(stringsAsFactors=FALSE)

#source("fun_v3.R")

if (! SYMALGEBRA) {
	library(polyAlgebra,quietly=TRUE,warn.conflicts=FALSE)
} else {
	library(gvector)
	library(symAlgebra,quietly=TRUE,warn.conflicts=FALSE)
}

source("bunch.R")
source("linemark.R")

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

c_table_decl = function(d, sizes=TRUE) {
	trim <- function (x) gsub("^\\s+|\\s+$", "", x)
	d = as.character(d)
	sel = grepl("\\[",d)
	if(any(sel)) {
		w = d[sel]
#		w = regmatches(w,regexec("([^[]*)\\[ *([^\\] ]*) *]",w))
		r = regexpr("\\[[^]]*\\]",w)
		w = lapply(1:length(r), function(i) {
			a_=w[i]
			c(a_,
				trim(substr(a_,1,r[i]-1)),
				trim(substr(a_,r[i]+1,r[i]+attr(r,"match.length")[i]-2))
			)
		})

		w = do.call(rbind,w)
		w = data.frame(w)
		w[,3] = as.integer(as.character(w[,3]))
		if (sizes) {
			w = by(w,w[,2],function(x) {paste(x[1,2],"[",max(x[,3])+1,"]",sep="")})
		} else {
			w = by(w,w[,2],function(x) {x[1,2]})
		}
		w = do.call(c,as.list(w))
	} else {
		w = c()
	}
	w = c(w,d[!sel])
	w
}


ifdef.global.mark = F
ifdef = function(val=F, tag="ADJOINT") {
	if ((!ifdef.global.mark) && ( val)) cat("\n#ifdef",tag,"\n");
	if (( ifdef.global.mark) && (!val)) cat("\n#endif //",tag,"\n");
	ifdef.global.mark <<- val
}


DensityAll = data.frame(parameter=logical(0))
Globals = data.frame()
Settings = data.frame()
ZoneSettings = data.frame()
Quantities = data.frame()
NodeTypes = data.frame()
Fields = data.frame()


AddDensity = function(name, dx=0, dy=0, dz=0, comment="", field=name, adjoint=F, group="", parameter=F,average=F) {
	if (any((parameter) && (dx != 0) && (dy != 0) && (dz != 0))) stop("Parameters cannot be streamed (AddDensity)");
	if (missing(name)) stop("Have to supply name in AddDensity!")
	if (missing(group)) group = name
	comment = ifelse(comment == "", name, comment);
	dd = data.frame(
		name=name,
		field=field,
		dx=dx,
		dy=dy,
		dz=dz,
		comment=comment,
		adjoint=adjoint,
		group=group,
		parameter=parameter,
		average=average
	)
	DensityAll <<- rbind(DensityAll,dd)
	for (d in rows(dd)) {
		AddField(name=d$field,
			dx=-d$dx,dy=-d$dy,dz=-d$dz,
			comment=d$comment,
			adjoint=d$adjoint,
			group=d$group,
			parameter=d$parameter,
			average=d$average,
		)
	}
}

AddField = function(name, stencil2d=NA, stencil3d=NA, dx=0, dy=0, dz=0, comment="", adjoint=F, group="", parameter=F,average=F) {
	if (missing(name)) stop("Have to supply name in AddField!")
	if (missing(group)) group = name
	comment = ifelse(comment == "", name, comment);
		d = data.frame(
			name=name,
			minx=min(dx,-stencil2d,-stencil3d,na.rm=T),
			maxx=max(dx, stencil2d, stencil3d,na.rm=T),
			miny=min(dy,-stencil2d,-stencil3d,na.rm=T),
			maxy=max(dy, stencil2d, stencil3d,na.rm=T),
			minz=min(dz,           -stencil3d,na.rm=T),
			maxz=max(dz,            stencil3d,na.rm=T),
			comment=comment,
			adjoint=adjoint,
			group=group,
			parameter=parameter,
			average=average
		)

		if (any(Fields$name == d$name)) {
			i = which(Fields$name == d$name)
			Fields$minx[i] <<- min(Fields$minx[i], d$minx)
			Fields$maxx[i] <<- max(Fields$maxx[i], d$maxx)
			Fields$miny[i] <<- min(Fields$miny[i], d$miny)
			Fields$maxy[i] <<- max(Fields$maxy[i], d$maxy)
			Fields$minz[i] <<- min(Fields$minz[i], d$minz)
			Fields$maxz[i] <<- max(Fields$maxz[i], d$maxz)
		} else {
			Fields <<- rbind(Fields, d)
		}
}


AddSetting = function(name,  comment, default=0, unit="1", adjoint=F, derived, equation, zonal=FALSE, ...) {
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
	if (zonal) {
		ZoneSettings <<- rbind(ZoneSettings,s)
	} else {
		Settings <<- rbind(Settings,s)
	}
}


AddGlobal = function(name, var, comment="", unit="1", adjoint=F, op="SUM", base=0.0) {
	if (missing(name)) stop("Have to supply name in AddGlobal!")
	if (missing(var)) var=name
	if (comment == "") {
		comment = name
	}
	if (!(op %in% c("SUM","MAX"))) stop("Operation (op) in AddGlobal have to be SUM or MAX")
	g = data.frame(
		name=name,
		var=var,
		comment=comment,
		unit=unit,
		adjoint=adjoint,
		op=op,
		base_value=base
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
		vector=vector,
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


Description = NULL
AddDescription = function(short, long=short) {
	if (! is.null( Description ) ) {
		stop("Adding descripition twice!")
	}
	Description <<- list(
		name=MODEL,
		short=short,
		long=long
	)
}

AddNodeType("BGK","COLLISION")
AddNodeType("MRT","COLLISION")
# AddNodeType("MR","COLLISION")
# AddNodeType("Entropic","COLLISION")
AddNodeType("Wall","BOUNDARY")
AddNodeType("Solid","BOUNDARY")
AddNodeType("WVelocity","BOUNDARY")
AddNodeType("WPressure","BOUNDARY")
AddNodeType("WPressureL","BOUNDARY")
AddNodeType("EPressure","BOUNDARY")
AddNodeType("EVelocity","BOUNDARY")
# AddNodeType("MovingWall","BOUNDARY")
# AddNodeType("Heater","ADDITIONALS")
# AddNodeType("HeatSource","ADDITIONALS")
# AddNodeType("Wet","ADDITIONALS")
# AddNodeType("Dry","ADDITIONALS")
# AddNodeType("Propagate","ADDITIONALS")
AddNodeType("Inlet","OBJECTIVE")
AddNodeType("Outlet","OBJECTIVE")
# AddNodeType("Obj1","OBJECTIVE")
# AddNodeType("Obj2","OBJECTIVE")
# AddNodeType("Obj3","OBJECTIVE")
# AddNodeType("Thermometer","OBJECTIVE")
AddNodeType("DesignSpace","DESIGNSPACE")

Stages=NULL

AddStage = function(name, main=name, load.densities=FALSE, save.fields=FALSE, no.overwrite=FALSE, fixedPoint=FALSE) {
	s = data.frame(
		name = name,
		main = main,
		adjoint = FALSE,
		fixedPoint=fixedPoint
	)
	sel = Stages$name == name
	if (any(sel)) {
		if (no.overwrite) return();
		s$index = Stages$index[sel]
		s$tag = Stages$tag[sel]
		Stages[sel,] <<- s
	} else {
		if (is.null(Stages)) {
			s$index = 1
		} else {
			s$index = nrow(Stages) + 1
		}
		s$tag = paste("S",s$index,sep="__")
		Stages <<- rbind(Stages,s)
	}
	if (is.character(load.densities)) {
		sel = load.densities %in% DensityAll$name
		if (any(!sel)) stop(paste("Unknown densities in AddStage:", load.densities[!sel]))
		load.densities = DensityAll$name %in% load.densities
	}
	if (is.logical(load.densities)) {
		if ((length(load.densities) != 1) && (length(load.densities) != nrow(DensityAll))) stop("Wrong length of load.densities in AddStage")
		if (nrow(DensityAll) > 0) {
			DensityAll[,s$tag] <<- load.densities
		} else {
			DensityAll[,s$tag] <<- logical(0);
		}
	} else stop("load.densities should be logical or character")

	if (is.character(save.fields)) {
		sel = save.fields %in% Fields$name
		if (any(!sel)) stop(paste("Unknown fields in AddStage:", save.fields[!sel]))
		save.fields = Fields$name %in% save.fields
	}
	if (is.logical(save.fields)) {
		if ((length(save.fields) != 1) && (length(save.fields) != nrow(Fields))) stop("Wrong length of save.fields in AddStage")
		Fields[,s$tag] <<- save.fields
	} else stop("save.fields should be logical or character in AddStage")
}

Actions = list()

AddAction = function(name, stages) {
	Actions[[name]] <<- stages
}

Objectives = list()

AddObjective = function(name, expr) {
	if (class(expr) == "gvector") {
		if (length(expr) == 1) {
			 expr = expr[[1]]
		} else {
			stop("Only one expression for an objective in AddObjective")
		}
	}
	if (! inherits(expr,"pAlg")) stop("Objective need to be a polyAlgebra expression")
	if (! is.character(name)) stop("Objective name need to be a string in AddObjective")
	Objectives[[name]] <<- expr
}

source("Dynamics.R") #------------------------------------------- HERE ARE THE MODEL THINGS

for (i in Globals$name) AddObjective(i,PV(i))

if (is.null(Description)) {
	AddDescription(MODEL)
}

if (!"Iteration" %in% names(Actions)) {
	AddAction(name="Iteration", stages=c("BaseIteration"))
}
if (!"Init" %in% names(Actions)) {
	AddAction(name="Init", stages=c("BaseInit"))
}
AllStages = do.call(c,Actions)

if (("BaseIteration" %in% AllStages) && (!"BaseIteration" %in% Stages$name)) {
	AddStage(main="Run", name="BaseIteration", load.densities=TRUE, save.fields=TRUE, no.overwrite=TRUE)
}
if (("BaseInit" %in% AllStages) && (!"BaseInit" %in% Stages$name)) {
	AddStage(main="Init", name="BaseInit", load.densities=FALSE, save.fields=TRUE, no.overwrite=TRUE)
}

if (any(duplicated(Stages$name))) stop ("Duplicated Stages' names\n")
ntag = paste("Stage",Stages$name,sep="_")
i = match(Stages$tag,names(DensityAll))
if (any(is.na(i))) stop("Some stage didn't load properly")
names(DensityAll)[i] = ntag
i = match(Stages$tag,names(Fields))
if (any(is.na(i))) stop("Some stage didn't load properly")
names(Fields)[i] = ntag
Stages$tag = ntag
#Stages = Stages[order(Stages$level),]
row.names(Stages)=Stages$name

for (n in names(Actions)) { a = Actions[[n]]
	if (length(a) != 0) {
		if (any(! a %in% row.names(Stages))) stop(paste("Some stages in action",n,"were not defined"))
		sel = Stages[a,"tag"]
		f = Fields[,sel,drop=F]
		s = apply(f,1,sum)
		if (any(s) > 1) {
			stop(paste("Field", Fields$name[s>1],"is saved more then once in Action",n))
		}
	} else {
		stop(paste("There is a empty Action:",n))
	}
}

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

if (NodeShiftNum > 16) {
	stop("NodeTypes exceeds short int")
} else {
	ZoneBits = 16 - NodeShiftNum
	ZoneShift = NodeShiftNum
	if (ZoneBits == 0) warning("No additional zones! (too many node types) - it will run, but you cannot use local settings")
	ZoneMax = 2^ZoneBits
#	ZoneRange = 1:ZoneMax
#	NodeTypes = rbind(NodeTypes,data.frame(
#		name=paste("SettingZone",ZoneRange,sep=""),
#		group="SETTINGZONE",
#		index=ZoneRange,
#		Index=paste("SettingZone",ZoneRange,sep=""),
#		value=(ZoneRange-1)*NodeShift,
#		mask=(ZoneMax-1)*NodeShift,
#		shift=NodeShiftNum
#	))
	NodeTypes = rbind(NodeTypes,data.frame(
		name="DefaultZone",
		group="SETTINGZONE",
		index=1,
		Index="DefaultZone",
		value=0,
		mask=(ZoneMax-1)*NodeShift,
		shift=NodeShiftNum
	))
	NodeShiftNum = 16
	NodeShift = 2^NodeShiftNum
}

if (any(NodeTypes$value >= 2^16)) stop("NodeTypes exceeds short int")

NodeTypes = rbind(NodeTypes, data.frame(
	name="None",
	group="NONE",
	index=1,
	Index="None",
	value=0,
	mask=0,
	shift=0
))

Node=NodeTypes$value
names(Node) = NodeTypes$name

i = !duplicated(NodeTypes$group)
Node_Group=NodeTypes$mask[i]
names(Node_Group) = NodeTypes$group[i]
Node_Group["ALL"] = sum(Node_Group)


Scales = data.frame(name=c("dx","dt","dm"), unit=c("m","s","kg"));

add.to.var.name = function(n,s) {
		n = as.character(n)
		sel = grepl("[[]", n)
		if (any(sel)) {
			n[sel] = sub("[[]",paste(s,"[",sep=""), n[sel])
		}
		if (any(!sel)) {
			n[!sel] = paste(n[!sel], s, sep="")
		}
		n
}


DensityAll$adjoint_name = add.to.var.name(DensityAll$name,"b")
DensityAll$tangent_name = add.to.var.name(DensityAll$name,"d")

Fields$adjoint_name = add.to.var.name(Fields$name,"b")
Fields$tangent_name = add.to.var.name(Fields$name,"d")

Fields$area = with(Fields,(maxx-minx+1)*(maxy-miny+1)*(maxz-minz+1))
Fields$simple_access = (Fields$area == 1)

if (ADJOINT==1) {

	for (s in rows(Settings)) {
		AddGlobal(
			name=paste(s$name,"_D",sep=""),
			var=paste(s$name,"b",sep=""),
			comment=paste("Gradient of objective with respect to [",s$comment,"]",sep=""),
			adjoint=T
		)
	}
	AddSetting(name="Descent",        comment="Optimization Descent", adjoint=T)
	AddSetting(name="GradientSmooth", comment="Gradient smoothing in OptSolve", adjoint=T)
	AddGlobal(name="AdjointRes", comment="square L2 norm of adjoint change", adjoint=T)
}
	for (g in rows(Globals)) if (! g$adjoint){
		AddSetting(
			name=paste(g$name,"InObj",sep=""),
			comment=paste("Weight of [",g$comment,"] in objective",sep=""),
			adjoint=T,
			zonal=T
		)
	}

DensityAll$nicename = gsub("[][ ]","",DensityAll$name)
Density   = DensityAll[! DensityAll$adjoint, ]
DensityAD = DensityAll[  DensityAll$adjoint, ]

Fields$nicename = gsub("[][ ]","",Fields$name)

Fields = bunch(Fields)

AddSetting(name="Threshold", comment="Parameters threshold", default=0.5)

AddGlobal(name="Objective",comment="Objective function");

Margin = data.frame(
	name = paste("block",1:27,sep=""),
	side = paste("side",1:27,sep=""),
	dx   = rep(-1:1,times=9),
	dy   = rep(rep(-1:1,times=3),each=3),
	dz   = rep(-1:1,each=9),
	command=paste("Margin",1:27)
)

Margin$sides = c(
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
			0L
		} else {
			w[is.na(w)] = 0L
			(w[1]+1L) + (w[2]+1L)*3L + (w[3]+1L)*9L + 1L
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

NonEmptyMargin = sapply(Margin, function(m) m$size != 0)
NonEmptyMargin = Margin[NonEmptyMargin]


Settings$FunName = paste("SetConst",Settings$name,sep="_")

#Dispatch = data.frame(
#	Globals=c(   "No",    "No",  "Globs",  "Obj",   "No",      "Globs",    "No",       "Globs",   "No",      "Globs"),
#	Action =c(   "No",  "Init",     "No",   "No",  "Adj",        "Adj",   "Adj",         "Adj",  "Opt",        "Opt"),
#	Stream =c(   "No",    "No",     "No",   "No",  "Adj",        "Adj",   "Adj",         "Adj",  "Opt",        "Opt"),
#	globals=c(  FALSE,   FALSE,     TRUE,   TRUE,  FALSE,         TRUE,   FALSE,          TRUE,  FALSE,         TRUE),
#	adjoint=c(  FALSE,   FALSE,    FALSE,  FALSE,   TRUE,         TRUE,    TRUE,          TRUE,   TRUE,         TRUE),
#	zeropar=c(  FALSE,   FALSE,    FALSE,  FALSE,  FALSE,        FALSE,    TRUE,          TRUE,   TRUE,         TRUE),
#	suffix =c(     "", "_Init", "_Globs", "_Obj", "_Adj", "_Globs_Adj", "_SAdj", "_Globs_SAdj", "_Opt", "_Globs_Opt")
#)
Dispatch = data.frame(
	Globals=c(   "No", "Globs",  "Obj",   "No",      "Globs",    "No",       "Globs",   "No",      "Globs"),
	Action =c(   "No",    "No",   "No",  "Adj",        "Adj",  "SAdj",        "SAdj",  "Opt",        "Opt"),
	Stream =c(   "No",    "No",   "No",  "Adj",        "Adj",   "Adj",         "Adj",  "Opt",        "Opt"),
	globals=c(  FALSE,    TRUE,   TRUE,  FALSE,         TRUE,   FALSE,          TRUE,  FALSE,         TRUE),
	adjoint=c(  FALSE,   FALSE,  FALSE,   TRUE,         TRUE,    TRUE,          TRUE,   TRUE,         TRUE),
	zeropar=c(  FALSE,   FALSE,  FALSE,  FALSE,        FALSE,    TRUE,          TRUE,   TRUE,         TRUE),
	suffix =c(     "","_Globs", "_Obj", "_Adj", "_Globs_Adj", "_SAdj", "_Globs_SAdj", "_Opt", "_Globs_Opt")
)
Dispatch$adjoint_ver = Dispatch$adjoint
Dispatch$adjoint_ver[Dispatch$Globals == "Obj"] = TRUE

p = expand.grid(x=seq_len(nrow(Dispatch)), y=seq_len(nrow(Stages)+1))
Dispatch = cbind(
	Dispatch[p$x,],
	data.frame(
		stage = c(FALSE,rep(TRUE,nrow(Stages))),
		stage_name  = c("Get", Stages$name),
		stage_index = c(0,Stages$index)
	)[p$y,]
)
sel = Dispatch$stage
Dispatch$suffix[sel] = paste("_", Dispatch$stage_name[sel], Dispatch$suffix[sel], sep="")

Globals = Globals[order(Globals$op),]


Consts = NULL
for (n in c("Settings","DensityAll","Density","DensityAD","Globals","Quantities","Scales","Fields","Stages","ZoneSettings")) {
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
Consts = rbind(Consts, data.frame(name="ZONE_SHIFT",value=ZoneShift))
Consts = rbind(Consts, data.frame(name="ZONE_MAX",value=ZoneMax))
Consts = rbind(Consts, data.frame(name="DT_OFFSET",value=ZoneMax*nrow(ZoneSettings)))
Consts = rbind(Consts, data.frame(name="GRAD_OFFSET",value=2*ZoneMax*nrow(ZoneSettings)))
Consts = rbind(Consts, data.frame(name="TIME_SEG",value=4*ZoneMax*nrow(ZoneSettings)))

offsets = function(d2=FALSE, cpu=FALSE) {
	def.cpu = cpu
	mw = PV(c("nx","ny","nz"))
	if2d3d = c(FALSE,FALSE,d2 == TRUE)
	one = PV(c(1L,1L,1L))
	bp = expand.grid(x=1:3,y=1:3,z=1:3)
	p = expand.grid(x=1:3*3-2,y=1:3*3-1,z=1:3*3)
	tab1 = c(1,-1,0)
	tab2 = c(0,-1,1)
	get_tab = cbind(tab1[bp$x],tab1[bp$y],tab1[bp$z],tab2[bp$x],tab2[bp$y],tab2[bp$z])
	sizes = c(one,mw,one)
	sizes[c(FALSE,FALSE,FALSE, if2d3d, FALSE,FALSE,FALSE)] = PV(1L)
	size  =  sizes[p$x]  * sizes[p$y]  * sizes[p$z]
	MarginNSize = PV(rep(0L,27))
	ret = lapply(Fields, function (f) 
	{
		mins = c(f$minx,f$miny,f$minz)
		maxs = c(f$maxx,f$maxy,f$maxz)
		tab1 = c(0,0,0,ifelse(mins > 0 & maxs > 0,-1,0),ifelse(maxs > 0,1,0))
		tab2 = c(ifelse(mins < 0,1,0),ifelse(maxs < 0 & mins < 0,-1,0),0,0,0)
		tab3 = c(mins<0,TRUE,TRUE,TRUE,maxs>0)
		put_tab = cbind(tab1[p$x],tab1[p$y],tab1[p$z],tab2[p$x],tab2[p$y],tab2[p$z])
		put_sel = tab3[p$x] & tab3[p$y] & tab3[p$z]
		mins = pmin(mins,0)
		maxs = pmax(maxs,0)
		nsizes = c(PV(as.integer(-mins)),one,PV(as.integer(maxs)))
		if (any(mins[if2d3d] != 0)) stop("jump in Z in 2d have to be 0")
		if (any(maxs[if2d3d] != 0)) stop("jump in Z in 2d have to be 0")
		nsize = nsizes[p$x] * nsizes[p$y] * nsizes[p$z]
		mSize = MarginNSize
		MarginNSize <<- mSize + nsize
		offset.p = function(positions,cpu) {
			positions[c(mins > -2, if2d3d, maxs < 2)] = PV(0L)
			if (cpu) {
			offset =  (positions[p$x] +
				  (positions[p$y] +
				  (positions[p$z]
					) * sizes[p$y] * nsizes[p$y]
					) * sizes[p$x] * nsizes[p$x]
					) * MarginNSize +
				  mSize
			} else {
			offset =   positions[p$x] +
				  (positions[p$y] +
				  (positions[p$z]
					) * sizes[p$y] * nsizes[p$y]
					) * sizes[p$x] * nsizes[p$x] +
				  mSize * size
			}
			offset
		}
		c(f,list(
			get_offsets = 
			function(w,dw,cpu=def.cpu) {
				tab1 = c(ifelse(dw<0,1,0),ifelse(dw<0,-1,0),0,0,0)
				tab2 = c(0,0,0,ifelse(dw>0,-1,0),ifelse(dw>0,1,0))
				tab3 = c(dw<0,TRUE,TRUE,TRUE,dw>0)
				get_tab = cbind(tab1[p$x],tab1[p$y],tab1[p$z],tab2[p$x],tab2[p$y],tab2[p$z])
				get_sel = tab3[p$x] & tab3[p$y] & tab3[p$z]
				offset = offset.p(c(w+PV(as.integer(dw)) - PV(as.integer(mins)),w+PV(as.integer(dw)),w+PV(as.integer(dw)) - mw),cpu=cpu)
				cond = c(w+PV(as.integer(dw)),mw-w-PV(as.integer(dw))-one)
				list(Offset=offset,Conditions=cond,Table=get_tab,Selection=get_sel)
			},
			put_offsets = 
			function(w,cpu=def.cpu) {
				offset = offset.p(c(w - mw - PV(as.integer(mins)),w,w),cpu=cpu)
				cond = c(w+PV(as.integer(-maxs)),mw-w+PV(as.integer(mins))-one)
				list(Offset=offset,Conditions=cond,Table=put_tab,Selection=put_sel)
			},
			fOffset=mSize*size
		))
	})
	class(ret) = "bunch"
	attr(ret,"cols") = names(ret[[1]])
	list(Fields=ret, MarginSizes=MarginNSize * size)
}

ret = offsets(cpu=FALSE)
Fields = ret$Fields

for (i in 1:length(Margin)) {
	Margin[[i]]$Size = ret$MarginSizes[i]
	if (! is.zero(Margin[[i]]$Size)) {
		 Margin[[i]]$size = 1L;
	} else {
		Margin[[i]]$size = 0L
	}
	Margin[[i]]$opposite_side = Margin[[28-i]]$side
}

NonEmptyMargin = sapply(Margin, function(m) m$size != 0)
NonEmptyMargin = Margin[NonEmptyMargin]


Enums = list(
	eOperationType=c("Primal","Tangent","Adjoint","Optimize","SteadyAdjoint"),
	eCalculateGlobals=c("NoGlobals", "IntegrateGlobals", "OnlyObjective", "IntegrateLast"),
	eModel=as.character(MODEL),
	eAction=names(Actions),
	eStage=c(Stages$name,"Get"),
	eTape = c("NoTape", "RecordTape")
)

AllKernels = expand.grid(
	Op=Enums$eOperationType,
	Globals=Enums$eCalculateGlobals[1:3],
	Model=Enums$eModel,
	Stage=Stages$name
#	Stage=Enums$eStage
)

AllKernels$adjoint = (AllKernels$Op %in% c("Adjoint","Opt"))
AllKernels$TemplateArgs = paste(AllKernels$Op, ",", AllKernels$Globals, ",", AllKernels$Stage)
AllKernels$Node = paste("Node_Run <", AllKernels$TemplateArgs, ">")


################################################################################

git_version = function(){f=pipe("git describe --always --tags"); v=readLines(f); close(f); v}
git_branch  = function(path=""){
	cmd = "git branch | sed -n '/\\* /s///p'";
	if (path != "") {
		cmd = paste("cd",path,";",cmd)
	}
	f = pipe(cmd);
	v = readLines(f);
	close(f);
	v
}

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
