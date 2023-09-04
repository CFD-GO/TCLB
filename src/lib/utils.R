rows = function(x) {
	rows_df= function(x) {
		if (nrow(x) > 0) {
			lapply(1:nrow(x),function(i) lapply(x,"[[",i))
		} else {
			list()
		}
	};
	switch( class(x),
		list       = x,
		data.frame = rows_df(x)
	)
}

table_from_text = function(text) {
	con = textConnection(text);
	tab = read.table(con, header=T);
	close(con);
	tab
}

c_table_decl = function(d, sizes=TRUE) {
	trim <- function (x) gsub("^\\s+|\\s+$", "", x)
	d = as.character(d)
	sel = grepl("\\[",d)
	if(any(sel)) {
		w = d[sel]
#		w = regmatches(w,regexec("([^[]*)\\[ *([^\\] ]*) *]",w))
		r = regexpr("\\[[^]]*\\]",w)
		w = lapply(1:length(r), function(i) {
			a_=w[i]
			c(a_,
				trim(substr(a_,1,r[i]-1)),
				trim(substr(a_,r[i]+1,r[i]+attr(r,"match.length")[i]-2))
			)
		})

		w = do.call(rbind,w)
		w = data.frame(w)
		w[,3] = as.integer(as.character(w[,3]))
		if (sizes) {
			w = by(w,w[,2],function(x) {paste(x[1,2],"[",max(x[,3])+1,"]",sep="")})
		} else {
			w = by(w,w[,2],function(x) {x[1,2]})
		}
		w = do.call(c,as.list(w))
	} else {
		w = c()
	}
	w = c(w,d[!sel])
	w
}


ifdef.global.mark = F
ifdef = function(val=F, tag="ADJOINT") {
	if ((!ifdef.global.mark) && ( val)) cat("\n#ifdef",tag,"\n");
	if (( ifdef.global.mark) && (!val)) cat("\n#endif //",tag,"\n");
	ifdef.global.mark <<- val
}
