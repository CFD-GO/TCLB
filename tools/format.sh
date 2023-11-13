#!/bin/bash

PP=$(dirname $0)

function format {
	clang-format --style=file:$PP/.clang-format \
	| sed -E 's/for[[:blank:]]*[(]([[:alpha:]_]*[[:blank:]]|)[[:blank:]]*([[:alnum:]_]+)[[:blank:]]*=[[:blank:]]*([[:alnum:]_]+)[[:blank:]]*;[[:blank:]]*([[:alnum:]_]+)[[:blank:]]*([<=]*)[[:blank:]]*([[:alnum:]_]+)[[:blank:]]*;[[:blank:]]*([[:alnum:]_]+)[+][+][[:blank:]]*\)/for (\1\2=\3; \4\5\6; \7++)/g' \
	| sed -E 's| ([*/]) |\1|g'
}

function formatRT {
	R -s -e 'rtemplate::RTtokenize()' | format | R -s -e 'rtemplate::RTtokenize(inv=TRUE)'
}

function chsum {
	sha256sum | cut -c 1-30
}

if test -z "$1"
then
	formatRT
else
	F=format
	if [[ "$1" =~ \.[rR][tT]$ ]]
	then
		F=formatRT
	fi
	if test -z "$2"
	then
		cat "$1" | $F
	else
		mkdir -p $PP/.format
		NAMESUM=$(echo "$1 $2" | sha256sum | cut -c 1-30)
		SUMFILE="$PP/.format/$NAMESUM"
		if test -f "$SUMFILE"
		then
			if sha256sum --check "$SUMFILE" >/dev/null
			then
				echo "Skipping $1"
				exit 0
			fi
		fi
		echo "Running $F on $1"
		cat "$1" | $F >tmp
		mv tmp "$2"
		sha256sum "$1" "$2" >"$SUMFILE"
	fi
fi
exit 0
