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
	log=output/$(echo $comment | sed 's|[ /\t]|.|g').log
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
		echo "FAILED"
		echo "Time: $(cat $log.time)s"
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
	TESTS=$(cd tests/$MODEL; ls *.test 2>/dev/null)
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
export PYTHONPATH="$PYTHONPATH:$PWD/tools/python"

function runline {
	CMD=$1
	R=$2
	G=$TEST_DIR/$R
	shift
	case $CMD in
	need)
		for i in "$@"
		do
			SRC=$TEST_DIR/$i
			if test -f "$SRC"
			then
				cp $SRC $i
			else
				echo "$i" not found;
				return -1;
			fi
		done
		echo "  copied $@"
		;;
	run) try "  running solver" "$@" ;;
	csvdiff) try "    checking $R (csvdiff)" $TCLB/tools/csvdiff -a "$R" -b "$G" -x 1e-10 -d Walltime ;;
	diff) try "    checking $R" diff "$R" "$G" ;;
	sha1) try "    checking $R (sha1)" sha1sum -c "$G.sha1" ;;
	*) echo "unknown: $CMD"; return -1;;
	esac
	return 0;
}

function testModel {
	for t in $TESTS
	do		
		name="${t%.test}"
		t="$name.test"
		TDIR="test-$MODEL-$name-$1"
		test -d "$TDIR" && rm -r "$TDIR"
		RESULT="OK"
		TCLB=".."
		TEST_DIR="../tests/$MODEL"
		if test -f "tests/$MODEL/$t"
		then
			echo "Running $name test..."
			mkdir -p $TDIR		
			while read line
			do
				if ! (cd $TDIR && runline $(eval echo $line))
				then
					RESULT="FAILED"
					break
				fi
			done < "tests/$MODEL/$t"
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

REPEAT=1
while test "$REPEAT" -le "$REPEATS"
do
	test "$REPEATS" -gt "1" && echo "############ repeat: $REPEAT ################"
	testModel $REPEAT
	REPEAT=$(expr $REPEAT + 1)
done

if test "x$GLOBAL" == "xOK"
then
	exit 0
else
	echo "Some tests failed"
	exit -1
fi
