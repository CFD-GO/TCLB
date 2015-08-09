
get.model.dirs = function(cmd) {
	f = pipe(paste(cmd, "sed -n 's|^src/\\([^/]*\\)/conf.mk$|\\1|gp'",sep=" | "))
	models= readLines(f)
	close(f)
	models
}

get.models = function() {
	M1 = get.model.dirs("git ls-files")
	M2 = get.model.dirs("ls src/*/conf.mk")
	M3 = union(M1,M2)
	
	Models = do.call(rbind, lapply(M3,function (m) {
		pt = paste("src",m,sep="/");
		cf = paste("src",m,"conf.mk",sep="/");
		e = new.env();
		e$ADJOINT=FALSE
		e$TEST=TRUE
		if (file.exists(cf)) {
			source(cf, local=e);
		} else {
			return(NULL)
		}
		if (is.numeric(e$ADJOINT)) e$ADJOINT = e$ADJOINT != 0
		if (is.logical(e$TEST)) e$TEST = ifelse(e$TEST,"test","no test")
		e$TEST = as.character(e$TEST)
		data.frame(name=m, adjoint=e$ADJOINT, test=e$TEST, path=pt, git=(m %in% M1), present=(m %in% M2))
	}))
	Models$test = factor(Models$test, levels=c("test","no test","compile only","can fail"))
	Models$experimental = Models$present & (! Models$git)
	if (any(is.na(Models$test))) stop("Wrong value of TEST in some conf.mk")
	if (any(! Models$present)) stop("Some model that is in git is not present in dir")
	Models
}
