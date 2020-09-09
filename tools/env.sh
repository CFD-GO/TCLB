#!/bin/bash

case $0 in
*env.sh) echo "This file should not be run as script, as it sets environment variables"; exit -1 ;;
*)
esac

for i in coreutils findutils gnu-sed
do
	if test -d "/usr/local/opt/$i/"
	then
		if test -d "/usr/local/opt/$i/libexec/gnubin/"
		then
			export PATH="/usr/local/opt/$i/libexec/gnubin/:$PATH"
		else
			echo "$i found but no 'gnubin' directory present"
			exit -1
		fi
	fi
done
