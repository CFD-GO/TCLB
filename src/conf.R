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
if (!exists("NEED_OFFSETS")) NEED_OFFSETS=TRUE
if (!exists("X_MOD")) X_MOD=0
if (!exists("CPU_LAYOUT")) CPU_LAYOUT=FALSE
if (!exists("plot_access")) plot_access=FALSE

memory_arr_cpu = CPU_LAYOUT
memory_arr_mod = X_MOD

# SYMALGEBRA=TRUE

options(stringsAsFactors=FALSE)

if (! SYMALGEBRA) {
	library(polyAlgebra,quietly=TRUE,warn.conflicts=FALSE)
} else {
	library(gvector)
	library(symAlgebra,quietly=TRUE,warn.conflicts=FALSE)
}

if (is.null(Options$autosym)) Options$autosym = FALSE

#source("linemark.R")

source("lib/utils.R")

DensityAll = data.frame(parameter=logical(0))
Globals = data.frame()
Settings = data.frame()
ZoneSettings = data.frame()
Quantities = data.frame()
NodeTypes = data.frame()
Fields = data.frame()
Stages = NULL

PartMargin=NA
permissive.access=FALSE

SetOptions = function(...) {
  args = list(...)
  optnames = c("permissive.access","PartMargin")
  idx = match(names(args), optnames)
  if (any(is.na(idx))) stop("Unknown options in SetOption: ", names(args)[is.na(idx)])
  if (any(duplicated(idx))) stop("Duplicated options in SetOption: ", names(args)[duplicated(idx)])
  for (i in seq_along(args)) {
    assign(optnames[idx[i]], args[[i]], envir = .GlobalEnv)
  }
}

AddDensity = function(name, dx=0, dy=0, dz=0, comment="", field=name, adjoint=F, group="", parameter=F, average=F, default=NA, sym=c("","",""), shift=NULL, ...) {
	if (missing(name)) stop("Have to supply name in AddDensity!")
	if (missing(group)) group = name
	if (length(sym) != 3) stop("sym provided to AddDensity have to be a vector of length 3");
	if (average && missing(default)) default=0;
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
		average=average,
		default=default,
		symX=sym[1],
		symY=sym[2],
		symZ=sym[3]
	)
	if (any((dd$parameter) & (dd$dx != 0) & (dd$dy != 0) & (dd$dz != 0))) stop("Parameters cannot be streamed (AddDensity)")
	DensityAll <<- rbind(DensityAll,dd)
	for (d in rows(dd)) {
		AddField(name=d$field,
			dx=-d$dx,dy=-d$dy,dz=-d$dz,
			comment=d$comment,
			adjoint=d$adjoint,
			group=d$group,
			parameter=d$parameter,
			average=d$average,
			sym=sym,
			shift=shift,
            ...
		)
	}
}

create_shift = function(type, ...) {
  ret = list(type=type, ...)
  class(ret) = "tclbshift"
  ret
}

no_shift = function() create_shift(type="no_shift")

single_shift = function(v) {
  if (v == 0 || is.na(v)) no_shift() else create_shift(type="single_shift", value=v)
}

convert_to_shift_list = function(n, x) {
  if (is.null(x)) x = list(NULL)
  if (identical(class(x), "list")) {
    if (length(x) == 1) {
      x = rep(x,n)
    } else if (length(x) != n) stop("Wrong length of list in 'shift' argument")
  } else if (is.numeric(x)) {
    if (length(x) == 1) {
      x = rep(x,n)
    } else if (length(x) != n) stop("Wrong length of list in 'shift' argument")
    x = lapply(x, single_shift)
  }
  if (! identical(class(x), "list")) stop("Shift needs to be convertable to list")
  x = lapply(x,function(x) if (is.null(x)) no_shift() else x)
  tp = sapply(x,function(x) identical(class(x),"tclbshift"))
  if (any(!tp)) stop("All elements of shift have to be of tclbshift class")
  x
}

AddField = function(name, stencil2d=NA, stencil3d=NA, dx=0, dy=0, dz=0, comment="", adjoint=F, group="", parameter=F,average=F, sym=c("","",""), shift=NULL,
                    optimise_for_static_access=TRUE, non.mandatory=FALSE) {
        shift = convert_to_shift_list(length(name), shift)
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
			average=average,
			symX=sym[1],
			symY=sym[2],
			symZ=sym[3],
			shift=I(shift),
            optimise_for_static_access=optimise_for_static_access,
			non.mandatory=non.mandatory
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
		  if (!is.null(Stages)) {
		    stop("It seems, that you added Field after Stage in Dynamics.R - this will not parse")
		  }
			Fields <<- rbind(Fields, d)
		}
}


AddSetting = function(name,  comment, default=0, unit="1", adjoint=F, derived, equation, zonal=FALSE, preload=TRUE, ...) {
	if (missing(name)) stop("Have to supply name in AddSetting!")
	if (any(unit == "")) stop("Empty unit in AddSetting not allowed")
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
		preload=preload,
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
	if (any(unit == "")) stop("Empty unit in AddGlobal not allowed")
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
	if (any(unit == "")) stop("Empty unit in AddQuantity not allowed")
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


## Description
read_a_file = function(file) {
	pot = c(file, paste(include.dir,file,sep="/"))
	sel = sapply(pot,file.exists)
	sel = which(sel)
	pot = pot[sel]
	if (length(pot) < 1) return(NULL)
	pot = pot[1]
	readLines(pot)
}

Description = NULL
AddDescription = function(short, long) {
	if (! is.null( Description ) ) {
		stop("Adding descripition twice!")
	}
	if (missing(long)) {
		long = read_a_file("Description.md")
		if (is.null(long)) long = short
	}
	Description <<- list(
		name=MODEL,
		short=short,
		long=long
	)
}


AddStage = function(name, main=name, load.densities=FALSE, save.fields=FALSE, read.fields=NA, can.overwrite=FALSE, default=FALSE, fixedPoint=FALSE, particle=FALSE, particle.margin) {
	s = data.frame(
		name = name,
		main = main,
		adjoint = FALSE,
		fixedPoint=fixedPoint,
		particle=particle,
		can.overwrite=can.overwrite
	)
	sel = Stages$name == name
	if (any(sel)) {
		if (default) return();
		stop("Two stages defined with the same name")
	}
        s$loadtag = paste0("LoadIn",s$name)
        s$savetag = paste0("SaveIn",s$name)
        s$readtag = paste0("ReadIn",s$name)
        Stages <<- rbind(Stages,s)
	if (! missing(particle.margin)) {
		if (! particle) stop("particle.margin declared in a stage, but particle=FALSE")
		PartMargin <<- max(PartMargin,particle.margin,na.rm=TRUE)
	}

	selection = function(tab,sel) {
		if (is.character(sel)) {
			if (any(!(sel %in% tab$name))) stop("load/save/read name not found in AddStage")
			sel = tab$name %in% sel
		}
		if (is.logical(sel)) {
			if (length(sel) == 1) sel = rep(sel, nrow(tab))
			if (length(sel) != nrow(tab)) stop("load/save/read invalid length in AddStage")
			return(sel)
		} else {
			stop("load/save/read invalid type in AddStage")
		}
	}

	sel = selection(DensityAll, load.densities)
	DensityAll[, s$loadtag] <<- sel
	sel = selection(Fields, save.fields)
	Fields[, s$savetag] <<- sel
	sel = selection(Fields, read.fields)
	Fields[, s$readtag] <<- sel
}

Actions = NULL

AddAction = function(name, stages) {
	a = data.frame(name=name, stages=I(list(stages)))
	Actions <<- rbind(Actions, a)
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

if (nrow(Fields) < 1) stop("The model has to have at least one Field/Density")

for (i in Globals$name) AddObjective(i,PV(i))

if (is.null(Description)) {
	AddDescription(MODEL)
}


if (Options$autosym) { ## Automatic symmetries
  symmetries = data.frame(symX=c(-1,1,1),symY=c(1,-1,1),symZ=c(1,1,-1))

  for (g in unique(DensityAll$group)) {
    D = DensityAll[DensityAll$group == g, ,drop=FALSE]
    for (d in rows(D)) {
      v = c(d$dx,d$dy,d$dz)
      for (s in names(symmetries)) if (d[[s]] == "") {
	if (all(v == 0)) {
		s_d = d
	} else {
          s_v = v * symmetries[,s]
          s_sel = (D$dx == s_v[1]) & (D$dy == s_v[2]) & (D$dz == s_v[3])
          if (sum(s_sel) == 0) stop("Could not find symmetry for density",d$name)
          if (sum(s_sel) > 1) stop("Too many symmetries for density",d$name)
          i = which(s_sel)
          s_d = D[s_sel,,drop=FALSE]
	}
        DensityAll[DensityAll$name == d$name,s] = s_d$name
        if (Fields[Fields$name == d$field,s] == "") Fields[Fields$name == d$field,s] = s_d$field
      }
    }
  }

  for (s in names(symmetries)) {
    sel = Fields[,s] == ""
    Fields[sel,s] = Fields$name[sel]
  }

  AddNodeType("SymmetryX_plus",  group="SYMX")
  AddNodeType("SymmetryX_minus", group="SYMX")
  AddNodeType("SymmetryY_plus",  group="SYMY")
  AddNodeType("SymmetryY_minus", group="SYMY")
  if (all(range(Fields$minz,Fields$maxz) == c(0,0))) {
	# we're in 2D
  } else {
	AddNodeType("SymmetryZ_plus",  group="SYMZ")
	AddNodeType("SymmetryZ_minus", group="SYMZ")
  }
}

if (!"Iteration" %in% Actions$name) {
	AddAction(name="Iteration", stages=c("BaseIteration"))
}
if (!"Init" %in% Actions$name) {
	AddAction(name="Init", stages=c("BaseInit"))
}
AllStages = unique(do.call(c,Actions$stages))

if (("BaseIteration" %in% AllStages) && (!"BaseIteration" %in% Stages$name)) {
	AddStage(main="Run", name="BaseIteration", load.densities=TRUE, save.fields=TRUE, default=TRUE)
}
if (("BaseInit" %in% AllStages) && (!"BaseInit" %in% Stages$name)) {
	AddStage(main="Init", name="BaseInit", load.densities=FALSE, save.fields=TRUE, default=TRUE)
}

if (any(duplicated(Stages$name))) stop ("Duplicated Stages' names\n")

row.names(Stages)=Stages$name

if (plot_access) {
	
	pa_fi = Fields
	pa_fi$index = seq_len(nrow(pa_fi))
	pa_fi = by(pa_fi, pa_fi$group, function(x) {x$groupsize = nrow(x); x})
	pa_fi = do.call(rbind,pa_fi)
	pa_fi = pa_fi[rev(seq_len(nrow(pa_fi))),]
	pa_fi$boxh = pmin(1,sqrt(3/pa_fi$groupsize))
	pa_fi$boxupper = cumsum(pa_fi$boxh)
	pa_fi$boxlower = c(0,pa_fi$boxupper[-length(pa_fi$boxupper)])
	pa_fi$boxmid = (pa_fi$boxupper + pa_fi$boxlower)/2
	#pa_sft = diff(pa_fi$boxmid)
	#sel = diff(as.integer(factor(pa_fi$group))) != 0
	#pa_sft[sel] = 1
	#pa_sft = cumsum(c(pa_fi$boxmid[1],pa_sft))
	#pa_sft = pa_sft - pa_fi$boxmid
	#pa_fi$boxupper = pa_fi$boxupper + pa_sft
	#pa_fi$boxlower = pa_fi$boxlower + pa_sft
	#pa_fi$boxmid   = pa_fi$boxmid   + pa_sft
	pa_fi = pa_fi[order(pa_fi$index),]
	pa_frange = range(pa_fi$boxupper,pa_fi$boxlower)
	pa_f = max(pa_fi$boxupper)
	pa_s = max(sapply(Actions,length))
	pa_ws = 10
	pa_scale = 0.2
	pa_ylab = max(strwidth(Fields$name,units="in")) + 0.3
	#pa_ylab = 2 # in
	pa_main = 0.8 # in
	pa_leg = 0.6 # in
	pdf("field_access.pdf",width=pa_ylab+pa_scale*(1+pa_ws*pa_s),height=pa_scale*diff(pa_frange)+pa_main+pa_leg)
	par(mai=c(pa_leg,pa_ylab,pa_main,0))
}
Actions$FunName = ifelse(Actions$name == "Iteration", "Iteration", paste0("Action_",Actions$name))

for (a in rows(Actions)) {
	if (length(a$stages) != 0) {
		if (!all(a$stages %in% Stages$name)) stop(paste("Some stages in action",a$name,"were not defined"))
		if (a$name == "Init") {
			bufin = rep(FALSE, nrow(Fields))
		} else {
			bufin = rep(TRUE, nrow(Fields))
		}
		bufout = rep(FALSE, nrow(Fields))
		first = TRUE
		pa_si = 0
		if (plot_access) {
			pa_s = length(a$stages)
			plot(NA,xlim=c(-0.5,pa_ws*pa_s+0.5),ylim=pa_frange,xaxt='n',yaxt='n',xlab="",ylab="",main=a$name,asp=1)
			legend(par('usr')[2], par('usr')[3], xpd=TRUE, yjust=1, xjust=1, ncol=2, cex=0.7, bty = "n", bg="white",
				legend = c("Previous iteration", "Newly written field", "Previously written field", "Density read", "Declared read access", "Implicit (undeclared) read access"),
				pch=c(15,15,15,NA,NA,NA),lty=c(NA,NA,NA,1,1,1),col=c("lightblue", "green","darkgreen","black","green","gray"))
			axis(2,at=pa_fi$boxmid,labels = Fields$name,las=1,gap.axis=0,cex.axis=0.6)
			abline(h=pa_fi$boxmid,col=8,lty=3)
			pa_col = rep("white",nrow(pa_fi))
			pa_col[bufin] = "lightblue"
			rect(-0.5,pa_fi$boxlower,0.5,pa_fi$boxupper,col=pa_col,border="darkblue")
			pa_sl = strwidth(Stages$name,units="in")
			pa_yscaling = par("pin")[2]/diff(par("usr")[3:4])
			pa_sl = max(pa_sl) / pa_yscaling
		}
		for (sn in a$stages) {
			pa_si = pa_si + 1
			s = Stages[Stages$name == sn,]
			ss = Fields[,s$savetag]
			sr = Fields[,s$readtag]
			sl = DensityAll[,s$loadtag]
			sl = Fields$name %in% unique(DensityAll$field[sl])
			sr[(!bufin) & is.na(sr)] = FALSE
			if (plot_access) {
				pa_col = rep("white",nrow(pa_fi))
				pa_col[bufout] = "darkgreen"
				pa_col[ss] = "green"
				rect(pa_ws*pa_si-0.5,pa_fi$boxlower,pa_ws*pa_si+0.5,pa_fi$boxupper,col=pa_col)
				rect(pa_ws*(pa_si-0.5)-0.7,pa_f/2-pa_sl/2-0.5,pa_ws*(pa_si-0.5)+0.7,pa_f/2+pa_sl/2+0.5,col="white")
				text(pa_ws*(pa_si-0.5),pa_f/2,labels=sn,srt=90)
				pa_a1x = pa_ws*(pa_si-1)+0.5
				pa_a1y = pa_fi$boxmid
				pa_a2x = pa_ws*(pa_si-0.5)-0.7
				pa_a2y = (pa_fi$boxmid/diff(pa_frange)-0.5)*pa_sl + pa_f/2
				pa_col = rep("white",nrow(pa_fi))
				pa_col[sr & (!is.na(sr))] = "green"
				pa_col[is.na(sr)] = "gray"
				pa_col[sl] = "black"
				pa_col[(!bufin) & (sr | sl)] = "red"
				sel = pa_col != "white"; if (any(sel)) segments(pa_a1x,pa_a1y[sel],pa_a2x,pa_a2y[sel],col=pa_col[sel])
				pa_a1x = pa_ws*(pa_si-0.5)+0.7
				pa_a1y = (pa_fi$boxmid/diff(pa_frange)-0.5)*pa_sl + pa_f/2
				pa_a2x = pa_ws*(pa_si-0)-0.5
				pa_a2y = pa_fi$boxmid
				pa_col = rep("white",nrow(pa_fi))
				pa_col[ss] = "black"
				pa_col[bufout & ss] = "red"
				sel = pa_col != "white"; if (any(sel)) segments(pa_a1x,pa_a1y[sel],pa_a2x,pa_a2y[sel],col=pa_col[sel])
			}
			sel = (!bufin) & (sr | sl)
			if (any(sel) && (! permissive.access)) stop("Reading fields [", paste(Fields$name[sel],collapse=", "),"] in stage '", sn,"' werent yet written in action '",a$name,"'")
			sel = bufout & ss
			if (any(sel) && (! s$can.overwrite) && (! permissive.access)) stop("Overwriting fields [", paste(Fields$name[sel],collapse=", "),"] in stage '", sn,"' that were written earlier in action '",a$name,"'")
			bufout = bufout | ss
			bufin = bufout
			first=FALSE
			Fields[,s$readtag] = sr
		}
		sel = (! bufout) & (! Fields$non.mandatory)
		if (any( sel ) && (! permissive.access)) stop("Fields [", paste(Fields$name[sel],collapse=", "),"] were not written in action '",a$name,"' (all fields need to be written*)\n*) in special cases you can mark a field as non.mandatory=TRUE")
	} else {
		stop(paste("There is a empty Action:",n))
	}
}

if (plot_access) {
	dev.off()
}

for (tag in Stages$readtag) {
	if (permissive.access) {
		Fields[,tag] = TRUE
	} else {
		Fields[is.na(Fields[,tag]),tag] = TRUE
	}
}

NodeShift = 1
NodeShiftNum = 0
if (nrow(NodeTypes) > 0) {
  NodeTypes = unique(NodeTypes)
  NodeTypes = do.call(rbind, by(NodeTypes,NodeTypes$group,function(tab) {
          n = nrow(tab)
          l = ceiling(log2(n+1))
          tab$index    = seq_len(n)
          tab$Index    = paste("NODE",tab$name,sep="_")
          tab$value    = NodeShift*(seq_len(n))
          tab$mask     = NodeShift*((2^l)-1)
          tab$max      = n
          tab$bits     = l
          tab$capacity = 2^l
          tab$shift = NodeShiftNum
          tab$groupIndex = paste("NODE",tab$group,sep="_")
          tab$save = TRUE
          NodeShift    <<- NodeShift * (2^l)
          NodeShiftNum <<- NodeShiftNum + l
          tab
  }))
} else {
  NodeTypes = data.frame()
}
FlagT = "unsigned short int"
FlagTBits = 16
if (NodeShiftNum > 14) {
	FlagT = "unsigned int"
	FlagTBits = 32
	if (NodeShiftNum > 30) {
		stop("NodeTypes exceeds 32 bits")
	}
}
ZoneBits = FlagTBits - NodeShiftNum
ZoneShift = NodeShiftNum
if (ZoneBits == 0) warning("No additional zones! (too many node types) - it will run, but you cannot use local settings")
ZoneMax = 2^ZoneBits
NodeTypes = rbind(NodeTypes,data.frame(
        name="DefaultZone",
        group="SETTINGZONE",
        index=1,
        Index="ZONE_DefaultZone",
        value=0,
        max=ZoneMax,
        bits=ZoneBits,
        capacity=ZoneMax,
        mask=(ZoneMax-1)*NodeShift,
        shift=NodeShiftNum,
        groupIndex = "NODE_SETTINGZONE",
        save = TRUE
))
NodeShiftNum = FlagTBits
NodeShift = 2^NodeShiftNum

if (any(NodeTypes$value >= 2^FlagTBits)) stop("NodeTypes exceeds size of flag_t")

#ALLBits = ZoneShift
#ALLMax = 2^ZoneShift
ALLBits = FlagTBits
ALLMax = 2^ALLBits
NodeTypes = rbind(NodeTypes,data.frame(
        name="None",
        group="NONE",
        index=1,
        Index="NODE_None",
        value=0,
        max=0,
        bits=0,
        capacity=0,
        mask=0,
        shift=0,
        groupIndex = "NODE_NONE",
        save = FALSE
))

NodeTypes = rbind(NodeTypes,data.frame(
        name="Clear",
        group="ALL",
        index=1,
        Index="NODE_Clear",
        value=0,
        max=ALLMax,
        bits=ALLBits,
        capacity=ALLMax,
        mask=(ALLMax-1),
        shift=0,
        groupIndex = "NODE_ALL",
        save = FALSE
))

NodeTypeGroups = unique(data.frame(
        name=NodeTypes$group,
        Index=NodeTypes$groupIndex,
        max=NodeTypes$max,
        bits=NodeTypes$bits,
        capacity=NodeTypes$capacity,
        mask=NodeTypes$mask,
        shift=NodeTypes$shift,
        save=NodeTypes$save
))

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

Fields$area = (Fields$maxx-Fields$minx+1)*(Fields$maxy-Fields$miny+1)*(Fields$maxz-Fields$minz+1)
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

AddGlobal(name="Objective",comment="Objective function");


tmp_c = Globals[Globals$op != "SUM",,drop=FALSE]; tmp_c = tmp_c[order(tmp_c$op),,drop=FALSE]
tmp_b = Globals[Globals$name == "Objective",,drop=FALSE]
tmp_a = Globals[Globals$op == "SUM" & Globals$name != "Objective",,drop=FALSE]

Globals = rbind(tmp_a,tmp_b,tmp_c)
SumGlobals = sum(Globals$op == "SUM")
ObjGlobalsIdx = which(Globals$name == "Objective")

if (any(Globals$op[seq_len(SumGlobals)] != "SUM")) stop("Something went wrong with ordering of globals")

	for (g in rows(Globals)[Globals$op == "SUM" & Globals$name != "Objective"]) if (! g$adjoint){
		AddSetting(
			name=paste(g$name,"InObj",sep=""),
			comment=paste("Weight of [",g$comment,"] in objective",sep=""),
			preload=FALSE,
			adjoint=T,
			zonal=T
		)
	}

DensityAll$nicename = gsub("[][ ]","",DensityAll$name)
Density   = DensityAll[! DensityAll$adjoint, ]
DensityAD = DensityAll[  DensityAll$adjoint, ]

Fields$nicename = gsub("[][ ]","",Fields$name)

AddSetting(name="Threshold", comment="Parameters threshold", default=0.5)

Margin = data.frame(
	name = paste("block",1:27,sep=""),
	side = paste("side[",1:27-1,"]",sep=""),
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


if (is.na(PartMargin)) PartMargin = 0.5

Consts = NULL
for (n in c("Settings","DensityAll","Density","DensityAD","Globals","Quantities","Scales","Fields","Stages","ZoneSettings","Actions")) {
	v = get(n)
	if (is.null(v)) v = data.frame()
	Consts = rbind(Consts, data.frame(name=toupper(n), value=nrow(v)));
	if (nrow(v) > 0) {
		v$index = seq_len(nrow(v))-1
		v$nicename = gsub("[][ ]","",v$name)
		v$Index = paste(" ",toupper(n), "_", v$nicename, " ", sep="")
		row.names(v) = v$name
		Consts = rbind(Consts, data.frame(name=v$Index, value=v$index));
		assign(n,v)
	}
	assign(n,v)
}

ret = merge(
	data.frame(name = paste0(Globals$name,"InObj"), glob.idx=Globals$index),
	data.frame(name = ZoneSettings$name, set.idx=ZoneSettings$index)
)
if (nrow(ret) > 0) {
	InObjOffset = ret$set.idx - ret$glob.idx
	if (any(InObjOffset != InObjOffset[1])) stop("Not all InObj offsets are the same. this should not happen")
	InObjOffset = InObjOffset[1]
} else {
	InObjOffset = 0
}
Consts = rbind(Consts, data.frame(name="IN_OBJ_OFFSET",value=InObjOffset))
Consts = rbind(Consts, data.frame(name="SUM_GLOBALS",value=SumGlobals))
Consts = rbind(Consts, data.frame(name="ZONE_SHIFT",value=ZoneShift))
Consts = rbind(Consts, data.frame(name="ZONE_MAX",value=ZoneMax))
Consts = rbind(Consts, data.frame(name="DT_OFFSET",value=ZoneMax*nrow(ZoneSettings)))
Consts = rbind(Consts, data.frame(name="GRAD_OFFSET",value=2*ZoneMax*nrow(ZoneSettings)))
Consts = rbind(Consts, data.frame(name="TIME_SEG",value=4*ZoneMax*nrow(ZoneSettings)))

is.power.of.two = function(x) { 2^floor(log(x)/log(2))-x != 0 }

if (is.power.of.two(memory_arr_mod)) stop("memory_arr_mod has to be a power of 2")

offsets = function() {
  mw = PV(c("nx","ny","nz"))
  one = PV(c(1L,1L,1L))
  p = expand.grid(x=1:3*3-2,y=1:3*3-1,z=1:3*3)
  sizes = c(one,mw,one)
  size  =  sizes[p$x]  * sizes[p$y]  * sizes[p$z]
  MarginNSize = PV(rep(0L,27))
  calc_functions = function(f) {
    mins = c(f$minx,f$miny,f$minz)
    maxs = c(f$maxx,f$maxy,f$maxz)
    tab1 = c(0,0,0,ifelse(mins == maxs & maxs > 0,-1,0),ifelse(maxs > 0,1,0))
    tab2 = c(ifelse(mins < 0,1,0),ifelse(maxs == mins & mins < 0,-1,0),0,0,0)
    tab3 = c(mins<0,TRUE,TRUE,TRUE,maxs>0)
    put_tab = cbind(tab1[p$x],tab1[p$y],tab1[p$z],tab2[p$x],tab2[p$y],tab2[p$z])
    put_sel = tab3[p$x] & tab3[p$y] & tab3[p$z]
    mins = pmin(mins,0)
    maxs = pmax(maxs,0)
    nsizes = c(PV(as.integer(-mins)),one,PV(as.integer(maxs)))
    nsize = nsizes[p$x] * nsizes[p$y] * nsizes[p$z]
    mSize = MarginNSize
    MarginNSize <<- mSize + nsize
    offset_p = function(positions) {
      positions[c(mins > -2, c(FALSE,FALSE,FALSE), maxs < 2)] = PV(0L)
      if (memory_arr_cpu) {
        offset =  (positions[p$x] +
                     (positions[p$y] +
                        (positions[p$z]
                        ) * sizes[p$y] * nsizes[p$y]
                     ) * sizes[p$x] * nsizes[p$x]
        ) * MarginNSize +
          mSize
      } else if (memory_arr_mod != 0) {
		positions__x = positions[p$x]
		positions_nx  = sizes[p$x]*nsizes[p$x]
		positions__x_mod = PV("((",ToC(positions__x),")&",memory_arr_mod-1,")")
		positions__x_div = (positions__x - positions__x_mod)*(1/memory_arr_mod)
		positions_nx_mod = PV(rep(memory_arr_mod, nrow(p)))
		positions_nx_div = positions_nx*(1/memory_arr_mod)
		sel = is.zero(positions_nx - PV(1L))
		dim(sel) = NULL
		positions__x_mod[sel] = positions__x[sel]
		positions__x_div[sel] = PV(0L)
		positions_nx_mod[sel] = positions_nx[sel]
		positions_nx_div[sel] = PV(1L)
		offset = positions__x_mod + positions_nx_mod*(
			positions[p$y] + sizes[p$y]*nsizes[p$y]*(
				positions__x_div + positions_nx_div*(
					positions[p$z]
				)
			)
		) + mSize * size
      } else {
		offset = positions[p$x] + sizes[p$x]*nsizes[p$x]*(
			positions[p$y] + sizes[p$y]*nsizes[p$y]*(
				positions[p$z]
			)
		) + mSize * size	
	  }
	  sel = is.zero(nsize)
	  dim(sel) = NULL
      offset[sel] = PV("NA")
      offset
    }
    list(get_offsets = 
      function(w,dw) {
		if (is.numeric(dw)) dw = PV(as.integer(dw))
		dw_neg = sapply(1:3, function(i) { if (is.numeric(dw[[i]])) dw[[i]]<0 else mins[i]<0 })
		dw_pos = sapply(1:3, function(i) { if (is.numeric(dw[[i]])) dw[[i]]>0 else maxs[i]>0 })
		tab1 = c(ifelse(dw_neg,1,0),ifelse(dw_neg,-1,0),0,0,0)
		tab2 = c(0,0,0,ifelse(dw_pos,-1,0),ifelse(dw_pos,1,0))
		tab3 = c(dw_neg,TRUE,TRUE,TRUE,dw_pos)
	mins = PV(as.integer(mins))
        get_tab = cbind(tab1[p$x],tab1[p$y],tab1[p$z],tab2[p$x],tab2[p$y],tab2[p$z])
        get_sel = tab3[p$x] & tab3[p$y] & tab3[p$z]
        offset = offset_p(c(w+dw - mins,w+dw,w+dw - mw))
        cond = c(w+dw,mw-w-dw-one)
        list(Offset=offset,Conditions=cond,Table=get_tab,Selection=get_sel)
      },
      put_offsets = 
      function(w) {
        offset = offset_p(c(w - mw - PV(as.integer(mins)),w,w))
        cond = c(w+PV(as.integer(-maxs)),mw-w+PV(as.integer(mins))-one)
        list(Offset=offset,Conditions=cond,Table=put_tab,Selection=put_sel)
      },
      fOffset=mSize*size
    )
  }
  ret = Fields
  ret$get_offsets = rep(list(NULL),nrow(ret))
  ret$put_offsets = rep(list(NULL),nrow(ret))
  ret$fOffset = rep(list(NULL),nrow(ret))
  for (idx in seq_len(nrow(ret))) {
      fun = calc_functions(ret[idx,])
      ret$get_offsets[[idx]] = fun$get_offsets
      ret$put_offsets[[idx]] = fun$put_offsets
      ret$fOffset[[idx]] = fun$fOffset
  }
  list(Fields=ret, MarginSizes=MarginNSize * size)
}

if (NEED_OFFSETS) {
    ret = offsets()
    Fields = ret$Fields
    for (i in seq_along(Margin)) {
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
}

BorderMargin = data.frame(
	name = c("x","y","z"),
	min  = c(min(0,Fields$minx),min(0,Fields$miny),min(0,Fields$minz)),
    max  = c(max(0,Fields$maxx),max(0,Fields$maxy),max(0,Fields$maxz))
)
BorderMargin$min[1] = 0  # We do not separate border in X direction.
BorderMargin$max[1] = 0



Enums = list(
	eOperationType=c("Primal","Tangent","Adjoint","Optimize","SteadyAdjoint"),
	eCalculateGlobals=c("NoGlobals", "IntegrateGlobals", "OnlyObjective", "IntegrateLast"),
	eModel=paste("model",as.character(MODEL),sep="_"),
	eAction=Actions$name,
	eStage=c(Stages$name,"Get"),
	eTape = c("NoTape", "RecordTape")
)

AllKernels = expand.grid(
	Op=Enums$eOperationType,
	Globals=Enums$eCalculateGlobals[1:3],
	Model=Enums$eModel,
	Stage=Stages$name
)

AllKernels$adjoint = (AllKernels$Op %in% c("Adjoint","Opt"))
AllKernels$TemplateArgs = paste(AllKernels$Op, ",", AllKernels$Globals, ",", AllKernels$Stage)
AllKernels$Node = paste("Node <", AllKernels$TemplateArgs, ">")


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
