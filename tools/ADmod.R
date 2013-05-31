f = file("Dynamics_b.c")

lines = readLines(f)
close(f)


bracel = gregexpr("[{]",lines)
bracer = gregexpr("[}]",lines)

a = sapply(bracel,function(x){sum(x>0)})
b = sapply(bracer,function(x){sum(x>0)})
a = cumsum(a-b)

a[a>1]=1
begins = which(diff(a)==1)+2

f = file("Dynamics_b.c_")
open(f,"wt")
pushi = grep("pushreal",lines)
popi = grep("popreal",lines)

begins = c(begins,length(lines))
alli = sort(c(pushi,popi,begins))
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
		l1 = sub("[ pushpopreal]*","",l);
		l1 = sub("_.*$","",l1);
		tp = switch(l1,"4"="float","8"="double");
		l1 = sub("[^(]*[(]","",l);
		l1 = sub("[)].*","",l1);
		var = l1;
		if (grepl("pushreal", l)) {
			idx = idx + 1
			name = paste(tmpname, idx, sep="_")
			if (idx > decl) {
				vars = c(vars, paste(tp,name,";"));
				decl = idx;
			}
			buf = c(buf, paste(name,"=", var, "; //",l));
		} else {
			var = sub("^[&]","",var);
			name = paste(tmpname, idx, sep="_")
			buf = c(buf, paste(var, " = ", name,"; //",l));
			idx = idx - 1;
		}
	}
	si = i;
}
close(f)