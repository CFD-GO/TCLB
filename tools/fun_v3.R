P = function(a){
	if (is.character(a)) {
		ret = data.frame(.M = 1)
		ret[,a] = 1;
		attr(ret,"var") = a
	} else if (is.numeric(a)) {
		ret = data.frame(.M = a)
#		if (a == 0) { ret = data.frame(.M = c()) }
		attr(ret,"var") = c()
	} else {
		stop("Unknow type in P()\n");
	}
	class(ret) = c("P","data.frame")
	ret
}	

scalar = function(v,a){
	
	if ("P" %in% class(a)) { a = as.factor(attr(a,"var")) }
	if (is.factor(a)) {
		ret = matrix(0, 1, nlevels(a)+1)
		ret = data.frame(ret)
		names(ret) = c(".M", levels(a))
		ret$.M = v;
		attr(ret,"var") = levels(a)
	} else {stop("a have to be a factor in scalar()");}
	class(ret) = c("P","data.frame")
	ret
}	


aggregate.P = function(p)
{
	if (nrow(p) > 0) {
                class(p) = "data.frame"
		if (length(attr(p,"var")) == 0) {
	                ret = p[1,,drop=FALSE]
			ret$.M = sum(p$.M)
		} else {
	                ret = aggregate(p[,".M",drop=FALSE], p[,attr(p,"var"),drop=FALSE], sum)
		}
                ret = ret[ret$.M != 0,,drop=FALSE]
                attr(ret,"var") = attr(p,"var")
                class(ret) = c("P","data.frame")
                ret
	}else {p}
}

rbind.P = function(p1,p2)
{
	if (nrow(p1) == 0) return(p2);
	if (nrow(p2) == 0) return(p1);
	class(p1) = "data.frame"
	class(p2) = "data.frame"
	col = names(p1)[!names(p1) %in% names(p2)]
	p2[,col] = 0;
	col = names(p2)[!names(p2) %in% names(p1)]
	p1[,col] = 0;
	ret = rbind(p1,p2)
	attr(ret,"var") = c(attr(p1,"var"),col)
	class(ret) = c("P","data.frame")
        ret
}

print.P = function(p)
{class(p) = "data.frame"; print(p)}

"+.P" <- function(p1,p2){
	p = rbind(p1,p2)
	p = aggregate(p)
	p
}

"*.P" <- function(p1,p2){
#	cat("-- *.P -----\n");
	if (! "P" %in%  class(p2)) {
		if (is.numeric(p2)) {
			if (length(p2) > 1) stop("P only multiply by numeric length 1\n")
			p1$.M = p1$.M * p2
			p1 = aggregate(p1)
			return(p1)
		}
		stop("Unknown type in P multyply\n");
	}
#	cat("----------------------------------------------------------------\n");
#	print(c(nrow(p1),nrow(p2)))
#	print(p1)
#	print(p2)

	if ((nrow(p1) != 0)&&(nrow(p2) != 0)) {               
                i = rep(1:nrow(p1),each =nrow(p2))
                j = rep(1:nrow(p2),times=nrow(p1))
                class(p1) = "data.frame"
                class(p2) = "data.frame"
		col = names(p1)[!names(p1) %in% names(p2)]
		p2[,col] = 0;
		col = names(p2)[!names(p2) %in% names(p1)]
		p1[,col] = 0;
                v = union(names(p1), names(p2))
		v = setdiff(v , ".M")
		if (length(v) != 0) {
	                p = p1[i,v,drop=FALSE] + p2[j,v,drop=FALSE]
        	        p$.M = p1$.M[i] * p2$.M[j]
		} else {
			p = P(p1$.M[i] * p2$.M[j])
		}
                attr(p,"var") = v
                class(p) = c("P","data.frame")
                p = aggregate(p)
	} else {
		if (nrow(p1) == 0) p = p1;
		if (nrow(p2) == 0) p = p2;
	}	
	p
}


PV = function(a){
	if (is.factor(a)) a = as.character(a)
	if (is.numeric(a) || is.character(a)) {
		ret = lapply(a,P)
	} else {
		if ("P" %in% class(a)) {
			ret = list(a)
		} else if ("PV" %in% class(a)) {
                        ret = a 
		} else {
			stop("unknown type in PV creation\n");
		}
	}
	class(ret) = c("PV")
	ret
}	

"+.PV" = function(p1,p2)
{
	if (is.numeric(p2)) { p2 = scalar(p2,p1[[1]]) }
	if ("P" %in% class(p2)) { p2 = PV(p2) }
	if ("PV" %in% class(p2)) {
		if (length(p1) == length(p2)) {
			ret = lapply(1:length(p1),function(i){
				p1[[i]] + p2[[i]]
			})
		} else {
			if (length(p2) == 1) {
				ret = lapply(1:length(p1),function(i){
					p1[[i]] + p2[[1]]
				})
			} else {
				stop("Non comforable PV vectors\n");
			}
		}
	} else {
		if (is.numeric(p2)) {
			stop("numeric + not yet implemented");
		} else { stop("Unknown type in +.PV");}
	}
        class(ret) = c("PV")
        ret
}

"-.PV" = function(p1,p2)
{
	p1 + p2 * (-1)
}

sum.PV = function(p,...){
	ret = p[[1]];
	if (length(p) > 1) {
	for (i in 2:length(p)) {
		ret = ret + p[[i]]
	}
	}
	PV(ret)
}

"*.PV" = function(p1,p2)
{
#	cat("--*-----\n");
	if (is.numeric(p1)) p1 = PV(p1)
	if (is.numeric(p2)) p2 = PV(p2)
	if (! "PV" %in% class(p1)) stop("Wrong type in *.PV")
	if (! "PV" %in% class(p2)) stop("Wrong type in *.PV")
	if (length(p1) == length(p2)) {
		ret = lapply(1:length(p1),function(i){
			p1[[i]] * p2[[i]]
		})
	} else {
		if (length(p2) == 1) {
			ret = lapply(1:length(p1),function(i){
				p1[[i]] * p2[[1]]
			})
		} else if (length(p1) == 1) {
			ret = lapply(1:length(p2),function(i){
				p1[[1]] * p2[[i]]
			})
		} else {
			stop("Non comforable PV vectors\n");
		}
	}
        class(ret) = c("PV")
        ret
}

"%%.PV" = function(p1,p2)
{
	if (! "PV" %in% class(p1)) {
		tmp = p2;
		p2 = t(p1);
		p1 = tmp;
	}
	if ("PV" %in% class(p2)) {
		stop("Cannot matrix multyply two PV vectors");
	} else {
		if (is.numeric(p2)) {
			p2 = as.matrix(p2)
			if (length(p1) == nrow(p2)) {
				i = 1;
				r1 = p1[i] * p2[i,]
				if (length(p1) > 1) for (i in 2:length(p1)){
					r2 = p1[i] * p2[i,]
					r1 = r1+r2
				}
				r1
			} else {
				stop("Non comforable matrix and PV vector\n");
			}
		} else { stop("Unknown type in +.PV");}
	}
}


"[.PV" = function(p,i)
{	
	ret = unclass(p)[i]
	class(ret) = c("PV")
	ret
}
 
ToC = function (x, ...) 
UseMethod("ToC")

ToC.PV = function(p, eq = TRUE, eqstring="=", float=TRUE, minimal=0.0)
{
	ret = sapply(p,function(x){ToC(x,float=float,minimal=minimal)})
	if (!is.null(names(ret)) && eq) {
		ret = paste(names(ret),eqstring,ret,";\n");
	}
	ret
}

ToC_row = function(x,float=TRUE,minimal=0.0)
{
	if (abs(x[".M"]) < minimal) x[".M"] = 0;
	if (x[".M"] != 0) {
		val = x[".M"]
		x[".M"] = abs(x[".M"])
		if (x[".M"] == 1) {
			ret = NULL
		} else {
			if (float) {
				ret = sprintf("%.10e",x[".M"])
			} else {
				ret = sprintf("%d",x[".M"])
			}	
#			ret = as.character(x[".M"])
		}
		v = names(x)
		for (i in 1:length(x))
		{
			vv = v[i]
			if (vv != ".M") {
				h = "";
				if (x[i] != 0) h = paste("pow(",vv,",",x[i],")")					
				if (x[i] == 1) h = vv					
				if (x[i] == 2) h = paste("(",vv,"*",vv,")",sep="")					
				if (x[i] == -1) h = paste("(1/",vv,")",sep="")					
				if (h != "") {
					if (is.null(ret)) {ret = h;} else {
						 ret = paste(ret,h,sep="*");
					}
				}
			}
		}
		if (is.null(ret)) {ret = "1"}
		ret = paste(ifelse(val>0," + "," - "), ret, sep="")
	} else {
		ret =""
	}
	ret
}

ToC.P = function(p,float=TRUE, minimal=0.0)
{
#	print(p)
	if (nrow(p) > 0) {
		ret = apply(p,1,function(x) {ToC_row(x,float=float,minimal=minimal)})
	} else { ret = "   0"; }
	ret = paste(ret,collapse="");
	if (substr(ret,2,2) == "+") substr(ret,2,2) = " ";
	if (ret == "") { ret = "   0"; }
	ret
}

Cassign = function(a,b)
{
	a = ToC(a);
	b = ToC(b);
	cat(paste(a," = ",b,";",sep=""),sep="\n")
}

"==.PV" = function(a,b)
{
	a = ToC(a,eq=FALSE);
	a = sub('[[:space:]]+$', '', a)
	a = sub('^[[:space:]]+', '', a)
	attr(b,"vvar") = a
	names(b) = a;
	b
}

der = function (x, ...) UseMethod("der")

der_row = function(x)
{
	val = x[".M"]
	v = names(x)
#	print(v)
	ret = NULL
	for (i in 1:length(x))
	{
		vv = v[i]
		if (vv != ".M") {
		if (x[i] > 0) {
			np = x;
			np[".M"] = np[".M"] * np[i]
			np[i] = np[i] - 1;
			np[der(vv)] = 1
#			print(np)
			ret = rbind(ret,np)
		}
		}
	}
#	print(ret)
	data.frame(ret)
}

der.P = function(p)
{
	class(p) = "data.frame"
	v = attr(p,"var")
	tmp = matrix(0, nrow(p), length(v))
	tmp = data.frame(tmp)
	names(tmp) = der(v)
	v = c(v,names(tmp))
	p = cbind(p,tmp)

	ret = apply(p,1,der_row)
#	print(ret)
	ret = do.call(rbind, ret)
#	print(ret)
	attr(ret,"var")=v
	class(ret) = c("P","data.frame")
#	print(ret)
        ret = aggregate(ret)
#	print(ret)
	ret
}

der.PV = function(x)
{
#	if ( is.null(names(x)) ) stop("x has no assignment in der.PV !")
	vout = attr(x,"vvar")
	vin = attr(x[[1]],"var")
#	print(vout)
#	print(vin)
	ret = list()
	for (out in vout)
	{
		p = x[[out]]
		v = vout
		tmp = matrix(0, nrow(p), length(v))
		tmp = data.frame(tmp)
		names(tmp) = der(v)
		p = cbind(p,tmp)
#		cat("-------------------------------------\n");
#		print(p)
#		cat("---------------and now --------------\n");
		for (v in vin)
		{
			j = (p[,v] > 0)
			if (any(j))
			{
				np = p[j,,drop=FALSE];
				np[,".M"] = np[,".M"] * np[,v]
				np[,v] = np[,v] - 1;
				np[,der(out)] = 1
#	                        print(np)
				if (is.null(ret[[v]])) {
					ret[[v]] = np
				} else {
					ret[[v]] = rbind(ret[[v]],np)
				}
			}
		}
	}
	for (i in names(ret))
	{
		p = ret[[i]];
#		print(p)
#		p = data.frame(p)
#		print(p)
		class(p) = c("P","data.frame")
		ret[[i]] = p
	}
	if (length(ret) != 0) names(ret) = der(names(ret))
	class(ret) = "PV";
	ret
}


der.character = function (x) {
	nx = sub("\\[","_d[",x)
	nx = ifelse( x == nx, paste(x,"_d",sep=""), nx)
	nx
}


#rbind.PV = function(x,y) {
#	x = unclass(x)
#	y = unclass(y)
#	ret = c(x,y)
#	class(ret) = "PV"
#	attr(ret,"vvar") = c(attr(x,"vvar"),attr(y,"vvar"))
#	ret
#}

rbind.PV = function(...) {
	args = list(...)
	args = lapply(args,PV)
	args = lapply(args,unclass)
	ret = do.call("c", args)
	class(ret) = "PV"
	vvar = sapply(args, function(x){attr(x,"vvar")})
	attr(ret,"vvar") = vvar
	ret
}


C = function(x,...) {cat(ToC(x,...), sep="");}
