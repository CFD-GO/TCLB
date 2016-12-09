f = pipe("grep '# *include' `find -regex '.*\\(c\\|cu\\|cpp\\|h\\|hpp\\)'` | sed -n 's/^\\([^:]*\\):[ \\t]*#[ \\t]*include[ \\t]*[\"<]\\(.*\\)[>\"]/\\1,\\2/gp'")
w = read.csv(f,col.names=c("file","dep"), stringsAsFactor=F);
w[,2] = paste0(sub("[^/]*$","",w[,1]),w[,2])
sel = sapply(w[,2],file.exists)
w = w[sel,]

resolve.path = function(x) sapply(strsplit(x,"/"),function(x) {
  x = x[x!="."];
  while (any(x[-1] == ".." & x[-length(x)] != "..")) {
    i = which(x[-1] == ".." & x[-length(x)] != "..")
    x = x[-c(i,i+1)]
  }
  paste(x,collapse="/")
})

w[,1] = resolve.path(w[,1])
w[,2] = resolve.path(w[,2])

dep = do.call(c,as.list(by(w,w$file,function(x) paste(x[1,1],paste(x[,2],collapse=" "),sep=" : ") )))
f=file("dep.mk")
if (length(dep) > 0) {
	cat(paste(dep,"\n\t@echo \"  DEP        $@ <-- $?\"\n\t@test -f $@ && touch $@\n\n",sep=""),sep="", file=f)
} else {
	cat("# no dep\n",file=f);
}
