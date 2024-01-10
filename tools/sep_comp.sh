#!/bin/bash

set -e

MODEL="$1"
if test -z "$MODEL"
then
    echo "usage: sep_comp.sh MODEL"
    exit -1
fi

make $MODEL/source

cd CLB/$MODEL/

for i in $(find -name "*.h" -or -name "*.hpp")
do
    echo "#include \"$i\"" >test.cpp
    if ! make test.o >$i.log 2>&1
    then
        echo "$i --- Bad"
    fi
    rm test.cpp
    test -f test.o && rm test.o
done
