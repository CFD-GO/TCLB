rows = function(x) {
        rows_df= function(x) {
                if (nrow(x) > 0) {
                        lapply(1:nrow(x),function(i) unclass(x[i,,drop=F]))
                } else {
                        list()
                }
        };
        ret = switch( class(x),
                list       = x,
                data.frame = rows_df(x)
        )
}

bunch = function(x,...) {
	arg = list(...)
	if (!missing(x)) {
		if (class(x) == "data.frame") {
			unclass(x)
			arg = c(arg,x)
		} else {
			arg = c(arg,x=x)
		}
	}	
	if (length(arg) > 0) {
		y = do.call(data.frame, arg)
	} else {
		stop("EMPTY at bunch")
	}
	n = names(y)
	y = rows(y)
	attr(y,"cols") = n
	class(y) = "bunch"
	y
}

names.bunch = function(x) {
	attr(x,"cols")
}

dim.bunch = function(x) {
	c(length(x),length(attr(x,"cols")))
}

"$.bunch" = function(x,name) {
	if ( name %in% attr(x,"cols") ) {
		sapply(x,function(x) { x[[name]] })
	} else {
		NULL
	}
}

"$<-.bunch" = function(x,name,value) {
	if (length(value) == length(x)) {
	} else if (length(value) == 1) {
		value = rep(value,length(x))
	} else stop("Wrong length of substitution in bunch $")
	if (length(x) < 1) {
		value = data.frame(value)
		names(value) = name
		bunch(value)
	} else {
		if ( ! (name %in% attr(x,"cols")) ) {
			attr(x,"cols") = c(attr(x,"cols"),name)
		}
		for (i in 1:length(x)) x[[i]][[name]] = value[[i]]
	}
	x
}

indexes.bunch = function(x,i) {
	if (length(x) < 1) {
		NULL
	} else {
		w = data.frame(row.names=1:length(x))
		for (j in names(x)) {
			h = get("$.bunch")(x,j)
			if (class(h) %in% c("numeric","character","factor","integer","logical"))
				w[,j]=h
		}
		(1:length(x))[eval(i,w)]
	}
}

"[.bunch" = function(x,i) {
	ni = indexes.bunch(x,substitute(i))
	y = unclass(x)[ni]
        class(y) = "bunch"
	attr(y,"cols") = attr(x,"cols")
        y
}

"[<-.bunch" = function(x,i,value) {
	if (class(value) == "list") {
		value = data.frame(value)
	}
	if (class(value) == "data.frame") {
		value = bunch(x=value)
	}
	if (!("bunch" %in% class(value)))
		stop("Wrong convertion in bunch [<-")
	if (!all(attr(value,"cols") %in% attr(x,"cols")))
		stop("Wrong names in bunch [<-")
	ni = indexes.bunch(x,substitute(i))
	if (length(ni) == 0) {
		x
	} else {
		if (length(ni) != length(value))
			stop("Wrong number of rows in bunch [<-");
		for (name in attr(value,"cols"))
			for (i in 1:length(value))
				x[[ni[i]]][[name]] = value[[i]][[name]]
		x
	}
}


print.bunch = function(x) {
	ret=do.call(rbind,lapply(x,data.frame))
	print(ret)
	ret
}




x = bunch(h=4,b=1:5)
