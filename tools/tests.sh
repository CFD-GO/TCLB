#!/bin/bash

function usage {
	echo "tests.sh [-h] [-r n] [-v] MODEL [TESTS]"
	if test "x$1" == "xhelp"
	then
		echo "         -h|--help        Help (this message)"
		echo "         -r|--repeat n    Repeat test n times"
		echo "         -v|--verbose     Verbose output"
		echo "         MODEL   The name of the model to test"
		echo "         TESTS   The list of test to run (optional)"
	fi
}

function try {
	comment="$1"
	mkdir -p output/
	log=output/$(echo $comment | sed 's/ /./g').log
	shift
	echo -n "$comment... "
	if env time -f "%e" -o $log.time "$@" >$log 2>&1
	then
		echo "OK ($(cat $log.time)s)"
		if $VERBOSE
		then
			cat $log
		fi
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

REPEATS=1
VERBOSE=false

while true
do
	case "$1" in
	-r|--repeat)
		REPEATS="$2"
		if ! test "$REPEATS" -gt 0
		then
			echo "Error: number of repeats have to be greater then 0"
			exit -1
		fi
		shift
		;;
	-v|--verbose)
		VERBOSE=true
		;;
	-h|--help)
		usage help
		exit 0;
		;;
	*)
		break;
	esac
	shift
done

MODEL=$1
if test -z "$MODEL"
then
	usage
	exit -1;
fi
shift

if ! test -f "CLB/$MODEL/main"
then
	echo "\"$MODEL\" is not compiled"
	echo "  run: make $MODEL"
	exit -1;
fi

if ! test -f "tests/README.md"
then
	echo "\"tests\" submodule is not checked out"
	echo "  run: git submodule init"
	echo "  run: git submodule update"
	exit -1
fi

if ! test -d "tests/$MODEL"
then
	echo "No tests for model $MODEL."
	echo "Exiting with no error."
	exit 0
fi

if test -z "$*"
then
	TESTS=$(cd tests/$MODEL; ls *.xml 2>/dev/null)
else
	TESTS="$*"
fi

if test -z "$TESTS"
then
	echo "No tests for model $MODEL \(WARNING: there is a directory tests/$MODEL !\)"
	echo "Exiting with error. Because I Can."
	exit -1
fi


GLOBAL="OK"
export PYTHONPATH="$PYTHONPATH:tools/python:tests/$MODEL"

function testModel {
	for t in $TESTS
	do
		name="${t%.xml}"
		t="$name.xml"
		RESULT="FAILED"
		
		if test -f "tests/$MODEL/$t"
		then
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
								sha1sum -c "$g" >/dev/null 2>&1 && R="OK"
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
								if test "x$EXT" == "xsha1"
								then
									cat $g
									pat=$(cat $g | sed 's/.*[ ][ ]*//')
									test -z "$pat" || sha1sum $pat
								fi
								RESULT="WRONG"
							fi
						else
							echo "Not found"
							RESULT="WRONG"
						fi
					done
				fi
			fi
		else
			echo "$t: test not found"
			RESULT="NOT FOUND"
		fi
		if ! test "x$RESULT" == "xOK"
		then
			echo " > Test \"$name\" returned: $RESULT"
			GLOBAL="FAILED"
		fi
	done
}

rm -r output-$MODEL-*/ 2>/dev/null || true

REPEAT=1
while test "$REPEAT" -le "$REPEATS"
do
	test "$REPEATS" -gt "1" && echo "############ repeat: $REPEAT ################"
	testModel
	mv -v output/ "output-$MODEL-$REPEAT"
	REPEAT=$(expr $REPEAT + 1)
done

if test "x$GLOBAL" == "xOK"
then
	exit 0
else
	echo "Some tests failed"
	exit -1
fi
