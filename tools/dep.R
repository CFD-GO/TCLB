#!/usr/bin/env Rscript

options(width=150)

setwd("src/")
f = pipe("grep -o '#[\\t ]*include[\\t ]*\"[^\"]*\"' `find -regex '.*\\(c\\|cu\\|cpp\\|h\\|hpp\\)\\(\\|\\.Rt\\)'` | sed -n 's/^\\([^:]*\\):#[ \\t]*include[ \\t]*\"\\([^\"]*\\)\"/\\1,\\2/gp'")
w = read.csv(f,col.names=c("file","dep"), stringsAsFactors=FALSE);


# function reducing the . and ..
resolve.path = function(x) sapply(strsplit(x,"/"),function(x) {
  x = x[! (x %in% c(".",""))];
  while (any(x[-1] == ".." & x[-length(x)] != "..")) {
    i = which(x[-1] == ".." & x[-length(x)] != "..")
    x = x[-c(i,i+1)]
  }
  paste(x,collapse="/")
})


# reduce the paths in w
w$file_o = resolve.path(w$file)
w$file = gsub(".Rt$","",w$file_o)
w$dep = resolve.path(paste(dirname(w$file),w$dep,sep="/"))

files = data.frame(name=unique(c(w$file, w$dep)),stringsAsFactors=FALSE)
row.names(files) = files$name
files$direct    = file.exists(files$name)
files$template  = file.exists(paste0(files$name,".Rt"))
files$configure = file.exists(paste0(files$name,".in"))
files$model     = grepl("templates/",files$name)
files$src       = files$direct + files$template + files$configure

if (any(files$src > 1)) stop("Too many sources for:", files$name[files$src > 1])
cat("No sources found for:")
files$no_src = files$src < 1
files$generated = ! files$direct
print(files$name[files$src < 1])

files$path = gsub("^templates","%",files$name)
#files$path = paste(ifelse(files$direct,"src","CLB"), files$path, sep="/")
files$path = paste("CLB", files$path, sep="/")


w = tapply(w$dep, w$file, function(x) x)

for (i in seq_along(w)) {
  x = w[[i]]
  l = -1
  while (length(x) != l) {
    l = length(x)
    x = unique(c(x,do.call(c,w[x])))
  }
  w[[i]] = x
}


nm = names(w)
nm = gsub("([.]cpp|[.]cu)$",".o", names(w))
nm = gsub("^templates","%",nm)
nm = paste("CLB",nm,sep="/")
sel = grepl("[.]o$",nm)
nm = nm[sel]
w = w[sel]

w = lapply(w, function(x) files[x,"path"])

sel_cu = grepl("[.]cu$",names(w))
deps = paste0(nm, " : ", files[names(w),"path"], "   ", sapply(w,paste,collapse=" "), " CLB/config.mk\n\t", ifelse(sel_cu,"$(compile_cu)","$(compile)"))

setwd("..")

writeLines(deps, con="CLB/dep.mk")

