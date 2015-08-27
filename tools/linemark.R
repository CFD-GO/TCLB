	linemark_lm = ""
	linemark_i = -2;
	linemark_ii = -1;
	linemark_iii = 1;
	txt_con = NULL
	txt_cont=FALSE

	number_of_quotes = function(txt) {
		ret = gregexpr("\"", txt)[[1]]
		if (length(ret) == 1) {
			if (ret == -1) return(0)
		}
		return(length(ret))
	}

	save_filename = ""
	save_line = -1
	save_in_quote = FALSE
	linemark_print = function(txt, filename, line) {
#		cat(paste0("[",paste0(txt,collapse=""),"] - ",filename,":",line,"\n"));
#		cat(paste0(txt,collapse=""));
#		print(txt)
#		txt=substr(txt,1,nchar(txt)-1)
#		cat(sprintf("%-90s // %s:%5d\n",txt,filename, line),sep="");
		for (t in txt) {
			if ((line != save_line) || (filename != save_filename)) {
				if (! save_in_quote) cat(paste0("#line ",line," \"",filename,"\"\n"))
				save_line <<- line
				save_filename <<- filename
			}
			q_num = number_of_quotes(t)
			if (q_num %% 2 == 1) save_in_quote <<- ! save_in_quote
			cat(t)
			save_line <<- save_line + 1
		}
	}


	left_txt = NULL
	left_filename = ""
	left_line = -1
	left_n = -1
	txt_filename = ""
	txt_line = -1
	linemark = function(i,filename,reset=FALSE) {
		if (! is.null(txt_con)) {
			cat("\n");
			sink()
			txt_txt = readLines(txt_con)
			close(txt_con)

			n = length(txt_txt)
			txt_txt[-n]=paste0(txt_txt[-n],"\n")
			if (! is.null(left_txt)) {
				left_txt[left_n] <<- paste0(left_txt[left_n],txt_txt[1])
				if (n>1) {
					linemark_print(left_txt, left_filename, left_line)
					left_txt <<- txt_txt[-1]
					left_n <<- n-1
					left_filename <<- txt_filename
					left_line <<- txt_line
				}
			} else {
					left_txt <<- txt_txt
					left_n <<- n
					left_filename <<- txt_filename
					left_line <<- txt_line
			}
			if (left_txt[left_n] == "") {
				if (left_n > 1) {
					linemark_print(left_txt[-left_n], left_filename, left_line)
				}
				left_txt <<- NULL
			}	
		}
		txt_filename <<- filename
		txt_line <<- i
		txt_con <<- file()
		sink(txt_con)
	}
