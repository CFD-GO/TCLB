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
begins = which(diff(a)==1)+1

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
		tp = switch(l1,
			"4"=list(type="float",array=FALSE),
			"8"=list(type="double",array=FALSE),
			"4array"=list(type="float",array=TRUE),
                        "8array"=list(type="double",array=TRUE),
                                 NULL);
		if (is.null(tp)) stop("Unknown type of push/pop: ",l);
		ar = tp$array
		tp = tp$type
		l1 = sub("[^(]*[(]","",l);
		l1 = sub("[)].*","",l1);
		if (ar) {
			l2 = sub("^[^,]*,[ ]*","", l1);
			ar_size = as.integer(l2);
			ar_dec = paste("[",ar_size,"]",sep="")
			ar_idx = paste("[",1:ar_size-1,"]",sep="")
			l1 = sub(",.*$","",l1);
		} else {
			ar_dec = ""
			ar_idx = ""
		}
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
					vars = c(vars, paste(tp," ",name,ar_dec,"; // ADmod.R: ",l,sep=""));
					decl = idx;
				}
				buf = c(buf, paste(name,ar_idx," = ", var,ar_idx, "; // ADmod.R: ",l,sep=""));
			} else if (grepl("lookreal", l)) {
				name = paste(tmpname, idx, sep="_")
				buf = c(buf, paste(var,ar_idx, " = ", name,ar_idx,"; // ADmod.R: ",l,sep=""));
			} else {
				name = paste(tmpname, idx, sep="_")
				buf = c(buf, paste(var,ar_idx, " = ", name,ar_idx,"; // ADmod.R: ",l,sep=""));
				idx = idx - 1;
			}
		}
	}
	si = i;
}
close(f)
