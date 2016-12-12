MODEL=$1

function usage {
	echo "tests.sh MODEL [TESTS]"
	exit -2
}

function try {
	comment=$1
	log=$(echo $comment | sed 's/ /./g').log
	shift
		echo -n "$comment... "
		if env time -f "%e" -o $log.time "$@" >$log 2>&1
		then
			echo "OK ($(cat $log.time)s)"
		else
			echo "FAILED ($(cat $log.time)s)"
			echo "----------------- CMD ----------------"
			echo $@
			echo "----------------- LOG ----------------"
			cat  $log
			echo "--------------------------------------"
			exit -1;
		fi
	return 0;
}


test -z "$MODEL" && usage

#if ! test -d "src/$MODEL"
#then
#	echo \"$MODEL\" is not a model
#	usage
#fi

if ! test -f "CLB/$MODEL/main"
then
	echo \"$MODEL\" is not compiled
	echo "  run: make $MODEL"
	exit -1;
fi

shift

if ! test -f "tests/README.md"
then
	echo \"tests\" submodule is not checked out
	exit -1
fi

if ! test -d "tests/$MODEL"
then
	echo No tests for model $MODEL.
	echo Exiting with no error.
	exit 0
fi

if test -z "$1"
then
	TESTS=$(cd tests/$MODEL; ls *.xml 2>/dev/null)
else
	echo "Running specific tests not yet implemented"
	exit -1
fi

if test -z "$TESTS"
then
	echo "No tests for model $MODEL \(WARNING: there is a directory tests/$MODEL !\)"
	echo "Exiting with error. Because I Can."
	exit -1
fi

GLOBAL="OK"
PP=$PYTHONPATH
for t in $TESTS
do
	name=${t%.*}
	RESULT="FAILED"
    export PYTHONPATH=$PP:tests/$MODEL
	if try "Running \"$name\" test" CLB/$MODEL/main "tests/$MODEL/$t"
	then
		RESULT="OK"
		RES=$(cd tests/$MODEL; find -name "${name}_*")
		if ! test -z "$RES"
		then
			for r in $RES
			do
				g=tests/$MODEL/$r
				echo -n " > Checking $r... "
				EXT=${r##*.}
				if test -f "$r" || [[ "x$EXT" == "xsha1" ]]
				then
					if ! test -f "$g"
					then
						echo "$g not found - this should not happen!"
						exit -123
					fi
					R="WRONG"
					COMMENT=""
					case "$EXT" in
					csv)
						COMMENT="(csvdiff)"
						tools/csvdiff -a "$r" -b "$g" -x 1e-10 -d Walltime >/dev/null && R="OK"
						;;
					sha1)
						COMMENT="(SHA1 checksum)"
                        sha1sum -c "$g" >> /dev/null && R="OK"
                        ;;
					*)
						diff "$r" "$g" >/dev/null && R="OK"
						;;
					esac

					if test "x$R" == "xOK"
					then
						echo "OK $COMMENT"
					else
						echo "Different $COMMENT"
						RESULT="WRONG"
					fi
				else
					echo "Not found"
					RESULT="WRONG"
				fi
			done
		fi
	fi
	if ! test "x$RESULT" == "xOK"
	then
		echo " > Test \"$name\" returned: $RESULT"
		GLOBAL="FAILED"
	fi
done

if test "x$GLOBAL" == "xOK"
then
	exit 0
else
	echo "Some tests failed"
	exit -1
fi
