MODEL=$1
if test -z "$MODEL"
then
	echo usage: tests.sh [model]
	exit -1
fi

if ! test -f "tests/README.md"
then
	echo \"tests\" submodule is not checked out
	exit -1
fi