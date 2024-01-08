#!/bin/bash

PP=$(dirname $0)

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

if ! test -f "tests/external/README.md"
then
	echo "\"tests/external\" submodule is not checked out"
	echo "  run: git submodule init"
	echo "  run: git submodule update"
	exit -1
fi

if ! test -d "tests/external/$MODEL"
then
	echo "No tests for model $MODEL."
	echo "Exiting with no error."
	exit 0
fi


TESTS_ARG="$*"
if test -z "$TESTS_ARG"
then
	TESTS_ARG="."
fi

TESTS=""
DIR="tests/external/$MODEL"
for t in $TESTS_ARG
do
	ADD=""
	if test -f "$DIR/$t"
	then
		ADD="$t"
	elif test -f "$DIR/$t.test"
	then
		ADD="$t.test"	
	elif test -d "$DIR/$t"
	then
		ADD="$(cd $DIR; ls $t/*.test 2>/dev/null)"
	else
		echo "Test not found: $t"
		exit -1
	fi
	TESTS="$TESTS $ADD"
done

if test -z "$TESTS"
then
	echo "No tests for model $MODEL (WARNING: there is a directory tests/external/$MODEL)"
	echo "Exiting with no error."
	exit 0
fi


GLOBAL="OK"
export PYTHONPATH="$PYTHONPATH:$PWD/tools/python"

function runline {
	CMD=$1
	shift
	case $CMD in
		run) try "running solver" "$@"; return $?;;
		fail) try "running solver (should fail)" '!' "$@"; return $? ;;
		csvconcatenate) try "concatenating csv files" $TCLB/tools/csvconcatenate "$@"; return $? ;;
	esac
	R=$1
	shift
	case $CMD in
		exists) try "checking $R (exists)" test -f "$R"; return $? ;;
		sha1) G="$R.sha1" ;;
		*) G="$R" ;;
	esac
	if test -f "$TEST_DIR/$1"
	then
		G="$1"
		shift
	fi
	G="$TEST_DIR/$G"
	if ! test -f "$G"
	then
		comment_fail "Requested file not found: $G"
		return -1
	fi
	case $CMD in
	need) try "copy needed file" cp "$G" "$R"; return $? ;;
	csvdiff) try "checking $R (csvdiff)" $TCLB/tools/csvdiff -a "$R" -b "$G" -x "${1:-1e-10}" -d ${2:-$CSV_DISCARD}; return $? ;;
	diff) try "checking $R" diff "$R" "$G"; return $? ;;
	sha1) try "checking $R (sha1)" sha1sum -c "$G.sha1"; return $? ;;
	pvtidiff) try "checking $R (pvtidiff)" $TCLB/CLB/$MODEL/compare "$R" "$G" "${1:-8}" ${2:-} ${3:-} ${4:-}; return $? ;; # ${2:-8} is { if $2 == "" then "8" else $2 }
	*) echo "unknown: $CMD"; return -1;;
	esac
	return 0;
}

function testModel {
	for t in $TESTS
	do		
		TEST="${t%.test}"
		t="$TEST.test"
		TEST=$(echo $TEST | sed 's|^[.]/||g' | sed 's|/|-|g')
		TDIR="test-$MODEL-$TEST-$1"
		test -d "$TDIR" && rm -r "$TDIR"
		RESULT="OK"
		TCLB=".."
		SOLVER="$TCLB/CLB/$MODEL/main"
		MODELBIN="$TCLB/CLB/$MODEL"
		TOOLS="$TCLB/tools"
		TEST_DIR="../tests/external/$MODEL"
		CAN_FAIL=false
		CSV_DISCARD=Walltime
		EXC_SH="$PP/etc/test.exceptions.sh"
		if test -f "$EXC_SH"
		then
			source "$EXC_SH"
		fi
		if test -f "tests/external/$MODEL/$t"
		then
			echo -e "\n\e[1mRunning $TEST test...\e[0m"
			mkdir -p $TDIR		
			while read -r -u 3 line
			do
				if ! (cd $TDIR && runline $(eval echo $line))
				then
					RESULT="FAILED"
					break
				fi
			done 3< "tests/external/$MODEL/$t"
		else
			echo "$t: test not found"
			RESULT="NOT FOUND"
		fi
#		echo -n "         Test \"$TEST\" returned:"
		if test "x$RESULT" == "xOK"
		then
			comment_ok   "$TEST test finished" "-----"
		else
			if $CAN_FAIL
			then
				comment_fail "$TEST test finished (can fail)" "-----"
			else			
				comment_fail "$TEST test finished" "-----"
				GLOBAL="FAILED"
			fi
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
