#!/bin/bash

PP=$(dirname $0)

R -s -e 'rtemplate::RTtokenize()' \
 | clang-format --style=file:$PP/.clang-format "$@" \
 | sed -E 's/for[[:blank:]]*[(]([[:alpha:]_]*[[:blank:]]|)[[:blank:]]*([[:alnum:]_]+)[[:blank:]]*=[[:blank:]]*([[:alnum:]_]+)[[:blank:]]*;[[:blank:]]*([[:alnum:]_]+)[[:blank:]]*([<=]*)[[:blank:]]*([[:alnum:]_]+)[[:blank:]]*;[[:blank:]]*([[:alnum:]_]+)[+][+][[:blank:]]*\)/for (\1\2=\3; \4\5\6; \7++)/g' \
 | sed -E 's| ([*/]) |\1|g' \
 | R -s -e 'rtemplate::RTtokenize(inv=TRUE)'
