#!/bin/bash

set -o pipefail

PP=$(dirname $0)
FORMATFILE="$PP/.clang-format"
PRINTSKIP=true

function format {
	clang-format --style=file:$FORMATFILE \
	| sed -E 's/for[[:blank:]]*[(]([[:alpha:]_]*[[:blank:]]|)[[:blank:]]*([[:alnum:]_]+)[[:blank:]]*=[[:blank:]]*([[:alnum:]_]+)[[:blank:]]*;[[:blank:]]*([[:alnum:]_]+)[[:blank:]]*([<=]*)[[:blank:]]*([[:alnum:]_]+)[[:blank:]]*;[[:blank:]]*([[:alnum:]_]+)[+][+][[:blank:]]*\)/for (\1\2=\3; \4\5\6; \7++)/g' \
	| sed -E 's| ([*/]) |\1|g'
}

function formatRT {
	R -s -e 'rtemplate::RTtokenize()' | format | R -s -e 'rtemplate::RTtokenize(inv=TRUE)'
}

function formatR {
        R -s -e "formatR::tidy_source('stdin', wrap=FALSE, args.newline=TRUE)"
}

function format_sel {
        case "$1" in
        *.[rR][tT])
                echo formatRT
                ;;
        *.[rR])
                echo formatR
                ;;
        *)
                echo format
                ;;
        esac
}

function format_to {
	F="$(format_sel "$1")"
	if test -z "$1" || test -z "$2"
	then
		echo "Something went wrong in format_to"
		exit 5
	fi
	mkdir -p $PP/.format
	NAMESUM=$(echo "$1 $2" | sha256sum | cut -c 1-30)
	SUMFILE="$PP/.format/$NAMESUM"
	if test -f "$SUMFILE"
	then
		if sha256sum --status --check "$SUMFILE"
		then
			$PRINTSKIP && echo "Skipping $1"
			return 0
		fi
	fi
	if [[ "$1" == "$2" ]]
	then
		echo "Running $F on $1"
	else
		echo "Running $F on $1 -> $2"
	fi
	cat "$1" | $F >tmp
	mv tmp "$2"
	sha256sum "$1" "$2" "$FORMATFILE" >"$SUMFILE"	
}

OUTFILE=""
OUTSAME=false
while test -n "$1"
do
	case "$1" in
	--help)
		echo ""
		echo "$0 [--all] [-o output] [-x] [files]"
		echo ""
		echo "  -o (--output) OUTPUT        : put the formated output in OUTPUT"
		echo "  -x (--overwrite)            : put the formated output in the same file"
		echo "  --all [FROM_DIR] [TO_DIR]   : format all source files in DIR"
		echo ""
		exit 0;
		;;
	-o|--output)
		shift
		OUTFILE="$1"
		;;
	-x|--overwrite)
		OUTSAME=true
		;;
	-a|--all)
		shift
		FROM_DIR="src"
		if ! test -z "$1"
		then
			if ! test -d "$1"
			then
				echo "$1: not a directory"
				exit 3
			fi
			FROM_DIR="$1"
			shift
		fi
		TO_DIR="$FROM_DIR"
		if ! test -z "$1"
		then
			if ! test -d "$1"
			then
				echo "$1: not a directory"
				exit 3
			fi
			TO_DIR="$1"
			shift
		fi
		if test "$FROM_DIR" == "$TO_DIR" && ! $OUTSAME
		then
			echo "Trying to format all file into the same directory. If you want to overwrite say '-x'"
			exit 4
		fi
		PRINTSKIP=false
		find "$FROM_DIR" | grep -E '[.](cpp|h|hpp|cu|c|R)([.]Rt|)$' | while read i
		do
			if test "$FROM_DIR" == "$TO_DIR"
			then
				j="$i"
			else
				j="$TO_DIR/${i#$FROM_DIR}"
			fi
			format_to "$i" "$j"
		done
		;;
	-p|--pipe)
		formatRT
		;;
	-*)
		echo "Uknown option $1"
		exit 1;
		;;
	*)
		INFILE="$1"
		if $OUTSAME && test -z "$OUTFILE"
		then
			OUTFILE="$INFILE"
		fi
		if test -z $OUTFILE
		then
			cat "$INFILE" | $(format_sel "$INFILE")
		else
			format_to "$INFILE" "$OUTFILE"
			OUTFILE=""
		fi
	esac
	shift
done

exit 0
