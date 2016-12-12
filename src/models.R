
get.model.dirs = function(cmd) {
	f = pipe(cmd)
	models= readLines(f)
	close(f)
	models
}

get.model.names = function(path) {
  ps = strsplit(path,"/")
  if (! all(sapply(ps, length) >= 2)) stop("Something is wrong with model paths (conf.mk in a wierd place)")
  pt = sapply(ps, function(x) paste(x[2:length(x)-1],collapse="/"))
  nm = sapply(ps, tail, 2)
  if (! all(nm[2,] == "conf.mk"))  stop("Something is wrong with model paths (inspect models.R)")
  nm = nm[1,]
  data.frame(name=nm, path=pt, conf=path)
}


get.models = function() {
	M1 = get.model.dirs("git ls-files | grep 'conf.mk$'")
	M2 = get.model.dirs("find models/ -name 'conf.mk'")
	M3 = union(M1,M2)
	Models = do.call(rbind, lapply(M3,function (m) {
		e = new.env();
		e$ADJOINT=FALSE
		e$TEST=TRUE
		if (file.exists(m)) {
			source(m, local=e);
		} else {
			return(NULL)
		}
		if (is.numeric(e$ADJOINT)) e$ADJOINT = e$ADJOINT != 0
		if (is.logical(e$TEST)) e$TEST = ifelse(e$TEST,"test","no test")
		e$TEST = as.character(e$TEST)
		data.frame(conf=m, adjoint=e$ADJOINT, test=e$TEST, git=(m %in% M1), present=(m %in% M2))
	}))
	Models=merge(Models, get.model.names(M3))
	Models$test = factor(Models$test, levels=c("test","no test","compile only","can fail"))
	Models$experimental = Models$present & (! Models$git)
	if (any(is.na(Models$test))) stop("Wrong value of TEST in some conf.mk")
#	if (any(! Models$present)) stop("Some model that is in git is not present in dir")
	Models
}
