#!/bin/bash

function usage {
	echo "tests.sh [-r n] [-v] MODEL [TESTS]"
	exit -2
}

function try {
	comment="$1"
	log=$(echo $comment | sed 's/ /./g').log
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

REPEAT=1

if [[ "x$1" == "x-r"  ]];
then
    REPEAT=$2
    if test $REPEAT -le 1;
    then
        REPEAT=2
    fi
    shift
    shift
fi

VERBOSE=false

if [[ "x$1" == "x-v"  ]];
then
    VERBOSE=true
    shift
fi

test -z "$1" && usage
MODEL=$1
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


function testModel {

GLOBAL="OK"
export PYTHONPATH="$PYTHONPATH:tools/python:tests/$MODEL"

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


if test $REPEAT -eq 1
then
    if test "x$GLOBAL" == "xOK"
    then
    	exit 0
    else
    	echo "Some tests failed"
    	exit -1
    fi
fi

}
echo "Tests are gonna be repeated $REPEAT times, output/ and Running*.log will be moved."
sleep 5
echo "Running..."

while test $REPEAT -gt 0
do
    echo "$REPEAT to go..."
    testModel
    #BELOW IS NOT PERFORMED ON LAST/ONLY TEST, testModel ends with exit X
    mv -v output output-$REPEAT
    mv -v Running* output-$REPEAT
    REPEAT=$(expr $REPEAT - 1)
done
