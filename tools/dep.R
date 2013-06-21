f = pipe("grep '# *include' *.c *.cu *.cpp *.h *.hpp | sed -n 's/^\\([^:]*\\):[ \\t]*#[ \\t]*include[ \\t]*[\"<]\\(.*\\)[>\"]/\\1,\\2/gp'")
w = read.csv(f,col.names=c("file","dep"), stringsAsFactor=F);
sel = sapply(w[,2],file.exists)
w = w[sel,]
dep = do.call(c,as.list(by(w,w$file,function(x) paste(x[1,1],paste(x[,2],collapse=" "),sep=" : ") )))
f=file("dep.mk")
if (length(dep) > 0) {
	cat(paste(dep,"\n\t@echo \"  DEP        $@ <-- $?\"\n\t@touch $@\n\n",sep=""),sep="", file=f)
} else {
	cat("# no dep\n",file=f);
}
