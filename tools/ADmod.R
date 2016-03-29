#!/usr/bin/env Rscript

library(optparse)
options <- list(
        make_option(c("-f","--file"), "store", default="", help="Input file", type="character"),
        make_option(c("-o","--out"), "store", default="", help="Output file", type="character"),
	make_option(c("-x","--fix"), "store", default="", help="variables to fix", type="character")
)

opt <- parse_args(OptionParser(usage="Usage: ADmod -f inputfile [-o outputfile]", options))

if (opt$file == "") stop("Input file not specified\nUsage: ADmod -f file\n");
if (opt$out == "") { opt$out = paste(opt$file, "_",sep="") }

fix = strsplit(opt$fix," ")[[1]]
cat("To fix:\n")
print(fix);


f = file(opt$file)

lines = readLines(f)
close(f)


bracel = gregexpr("[{]",lines)
bracer = gregexpr("[}]",lines)

a = sapply(bracel,function(x){sum(x>0)})
b = sapply(bracer,function(x){sum(x>0)})
a = cumsum(a-b)

a[a>1]=1
begins = which(diff(a)==1)+2

f = file(opt$out)
open(f,"wt")
pushi = grep("pushreal",lines)
looki = grep("lookreal",lines)
popi = grep("popreal",lines)

begins = c(begins,length(lines))
alli = sort(c(pushi,popi,begins,looki))
idx = 0
tmpname = "keep";
si = 0
buf = c()
vars = c()
decl = 0
for (i in alli) {
	if (i-si > 1) {
		buf = c(buf, lines[(si+1):(i-1)] );
	}
	if (i %in% begins) {
		buf = c(buf, lines[i] );
#		cat(vars,sep="\n")
#		cat(buf,sep="\n")
		if (length(vars) > 0) writeLines(text=vars,con=f)
		writeLines(text=buf,con=f)

		vars = c()
		buf = c()
		decl = 0;
	} else {
		l = lines[i]
		l1 = sub("[ pushpopreallook]*","",l);
		l1 = sub("[_]?\\(.*$","",l1);
		tp = switch(l1,"4"="float","8"="double", "unknown");
		l1 = sub("[^(]*[(]","",l);
		l1 = sub("[)].*","",l1);
		var = l1;
		var = sub("^[&]","",var);
		if (var %in% fix) {
			cat("var: ",var," ----- fixed\n");
			buf = c(buf, paste("//",l));
		} else {
			if (grepl("pushreal", l)) {
				idx = idx + 1
				name = paste(tmpname, idx, sep="_")
				if (idx > decl) {
					vars = c(vars, paste(tp,name,";"));
					decl = idx;
				}
				buf = c(buf, paste(name,"=", var, "; //",l));
			} else if (grepl("lookreal", l)) {
				name = paste(tmpname, idx, sep="_")
				buf = c(buf, paste(var, " = ", name,"; //",l));
			} else {
				name = paste(tmpname, idx, sep="_")
				buf = c(buf, paste(var, " = ", name,"; //",l));
				idx = idx - 1;
			}
		}
	}
	si = i;
}
close(f)
