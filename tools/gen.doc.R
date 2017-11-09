library(yaml)

options(error = function() traceback())

dat = yaml.load_file("doc/elements.yaml")

type.link = function(n) {
  if (is.null(dat$types[[n]])) {
    stop("no such element")
  } else {
    d = dat$types[[n]]
    paste("[",d$name,"](",d$filename,")",sep="")
  }
}

element.link = function(n)
  sapply(n,function(n) {
  if (is.null(dat[[n]])) {
    stop("no such element")
  } else {
    d = dat[[n]]
    if (is.null(d$type)) {
      paste("[",d$name,"](",d$name,")",sep="")
    } else {
      dt = dat$types[[d$type]]
      paste("[<code>&lt;",d$name,"/&gt;</code>](",dt$filename,"#",tolower(d$name),")",sep="")
    }
  }
})

for (n in names(dat$types)) {
  d = dat$types[[n]]
  if (is.null(d$name)) d$name = n
  d$filename = paste(gsub(" ","-",d$name),"md",sep=".")
  dat$types[[n]] = d
}

for (n in names(dat)) if (n != "types") {
  d = dat[[n]]
  if ("type" %in% names(d)) {
    if (d$type %in% names(dat$types)) {
      cat("type: ",d$type,"\n")
      dt = dat$types[[d$type]]
      for (i in names(dt)) if (i != "name") d[[i]] = c(dt[[i]], d[[i]])
      dat$types[[d$type]]$ofthistype = c(dt$ofthistype, n)
    } else {
      stop("unknown type", d$type)
    }
  }
  if (is.null(d$name)) d$name = n
  dat[[n]] = d
}

element.link("VTK")
type.link("geom")


for (nt in names(dat$types)) {
t = dat$types[[nt]]
fn = t$filename
cat("----------",nt,t$name,"-------------\n");
fn = paste("wiki/xml/",fn,sep="");
cat(fn,"\n");
sink(fn);
cat("# ",t$name,"\n");
cat(t$comment);
for (n in t$ofthistype){
  d = dat[[n]]
  cat("## ",n,"\n\n")
  if (is.null(d$example)) {
    d$example = paste("<",n," .../>\n",sep="");
  }
  if (!is.null(d$example)) {
    i = length(d$example)
    d$example[i] = gsub("([^\n])$","\\1\n",d$example[i])
    cat("```xml\n");
    cat(d$example);
    cat("```\n\n");
  }
  cat(d$comment,"\n")
  if (!is.null(d$children)) {
    ret = sapply(d$children, function(k) {
      if (is.character(k)) {
        ret = element.link(k)
      } else if (!is.null(k$type)) {
        ret = type.link(k$type)
        dt = dat$types[[k$type]]
        if (!is.null(dt$ofthistype)){
          ret = paste(ret," (",paste(element.link(dt$ofthistype),collapse=", "),")",sep="")
        }
        ret
      } else stop("what?:", k)
    })
    cat("Possible children:", paste(ret,collapse=", "),"\n\n")
  }
  if (!is.null(d$attr)) {
    cat("\n| Attribute | Comment | Value |\n");
    cat("| --- | --- | --- |\n");
    for (a in d$attr) {
	if (!is.null(a$name)) {
	  if (is.character(a$name)) {
	    name = a$name
	  } else {
	    name = "Unknown"
	  }
	  comment = a$comment
	  if (is.null(a$val)) {
	    val = ""
	  } else if (! is.null(a$val["unit"])) {
	    val = paste("Value with unit (",a$val["unit"],")",sep="")
	  } else if (! is.null(a$val$numeric)) {
	    val = paste("Numeric (",a$val$numeric,")",sep="")
	  } else if (! is.null(a$val$list)) {
	    val = paste("Comma separated list of elements from:",sep="")
	  } else if (! is.null(a$val$select)) {
	    val = paste("Select from:", paste(a$val$select,collapse=", "))
	  } else {
	    val = "Unknown type"
	  }

	  cat("| `",name,"=` | ", comment, " | ", val," |\n",sep="");
	  
	}
    }
    cat("\n");
  }
}
sink()
}

