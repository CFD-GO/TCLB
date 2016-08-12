#!/bin/bash



MODEL=$1


files=$@ 
for f in $files
do
		EXT=${f##*.}
        if test "x$EXT" != "x" && test -f $f
        then
        g=$(basename $f)
		case "$EXT" in
			vti)
				
                sha1sum "$f" > ./tests/$MODEL/output/$g.sha1
                echo "SHA: $f"
             ;;
			*)
			    cp -v $f ./tests/$MODEL/output/$g
				;;
			esac
        fi
done
