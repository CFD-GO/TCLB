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

function comment_wait {
#       echo -ne "[      ] $1\r"
        printf   "[      ] %-70s %6s" "$1" "$2"
}
function comment_ok {
#       echo -e "[\e[92m  OK  \e[0m] $1"
        printf  "\r[\e[92m  OK  \e[0m] %-70s %6s\n" "$1" "$2"
}
function comment_fail {
#       echo -e "[\e[91m FAIL \e[0m] $1"
        printf  "\r[\e[91m FAIL \e[0m] %-70s %6s\n" "$1" "$2"
}

function try {
	comment="$1"
	shift
	mkdir -p output/
	log=$(echo $comment | sed 's|[ /\t]|.|g').log
	comment_wait "$comment"
	NEG=false
	if test "x$1" == 'x!'
	then
		NEG=true
		shift
	fi
	if env time --quiet -f "%e" -o $log.time "$@" >$log 2>&1
	then
		RES=true
	else
		RES=false
	fi
	if test $NEG != $RES
	then
		comment_ok "$comment" "$(cat $log.time)s"
		if $VERBOSE
		then
			(
				echo "----------------  LOG  ---------------"
				cat  $log
				echo "--------------------------------------"
			) | sed 's|^|         |'
		fi
	else
		comment_fail "$comment"
		(
			echo "----------------  CMD  ---------------"
			echo $@
			echo "----------------  TIME ---------------"
			echo "Time: $(cat $log.time)s"
			echo "----------------  LOG  ---------------"
			cat  $log
			echo "--------------------------------------"
		) | sed 's|^|         |'
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
	echo "No tests for model $MODEL (WARNING: there is a directory tests/$MODEL)"
	echo "Exiting with no error."
	exit 0
fi


GLOBAL="OK"
export PYTHONPATH="$PYTHONPATH:$PWD/tools/python"

function runline {
	CMD=$1
	shift
	R=$1
	G=$TEST_DIR/$R
	case $CMD in
	need) 
		comment_wait "copy $@"
		for i in "$@"
		do
			SRC=$TEST_DIR/$i
			if test -f "$SRC"
			then
				cp $SRC $i
			else
				comment_fail "copy $@"
				echo "         $i not found"
				return -1;
			fi
		done
		comment_ok "copy $@"
		;;
	run) try "running solver" "$@" ;;
	fail) try "running solver" '!' "$@" ;;
	csvdiff) try "checking $R (csvdiff)" $TCLB/tools/csvdiff -a "$R" -b "$G" -x 1e-10 -d Walltime ;;
	diff) try "checking $R" diff "$R" "$G" ;;
	sha1) try "checking $R (sha1)" sha1sum -c "$G.sha1" ;;
	pvtidiff) try "checking $R (pvtidiff)" $TCLB/CLB/$MODEL/compare "$R" "$G" "${2:-8}" ;; # ${2:-8} is { if $2 == "" then "8" else $2 }
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
		SOLVER="$TCLB/CLB/$MODEL/main"
		TEST_DIR="../tests/$MODEL"
		if test -f "tests/$MODEL/$t"
		then
			echo -e "\n\e[1mRunning $name test...\e[0m"
			mkdir -p $TDIR		
			while read -r -u 3 line
			do
				if ! (cd $TDIR && runline $(eval echo $line))
				then
					RESULT="FAILED"
					break
				fi
			done 3< "tests/$MODEL/$t"
		else
			echo "$t: test not found"
			RESULT="NOT FOUND"
		fi
#		echo -n "         Test \"$name\" returned:"
		if test "x$RESULT" == "xOK"
		then
			comment_ok   "$name test finished" "-----"
		else
			comment_fail "$name test finished" "-----"
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
