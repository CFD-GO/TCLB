#!/bin/bash

set -o pipefail

PP=$(dirname $0)
THISSCRIPT="$0"
FORMATFILE="$PP/.clang-format"
PRINTSKIP=true
SKIP_ON_KEEP=false

function formatCPP {
    FOPT=""
    if ! test -z "$FORMATFILE"; then
        FOPT="$FOPT --style=file:$FORMATFILE"
    fi
    if ! test -z "$ASSUMEFILE"; then
        FOPT="$FOPT --assume-filename=$ASSUMEFILE"
    fi
    clang-format $FOPT |
        sed -E 's/for[[:blank:]]*[(]([[:alpha:]_]*[[:blank:]]|)[[:blank:]]*([[:alnum:]_]+)[[:blank:]]*=[[:blank:]]*([[:alnum:]_]+)[[:blank:]]*;[[:blank:]]*([[:alnum:]_]+)[[:blank:]]*([<=]*)[[:blank:]]*([[:alnum:]_]+)[[:blank:]]*;[[:blank:]]*([[:alnum:]_]+)[+][+][[:blank:]]*\)/for (\1\2=\3; \4\5\6; \7++)/g' |
        sed -E 's| ([*/]) |\1|g'
}

function formatRT {
    R -s -e 'rtemplate::RTtokenize()' | $@ | R -s -e 'rtemplate::RTtokenize(inv=TRUE)'
}

function formatR {
    #R -s -e "formatR::tidy_source('stdin', wrap=FALSE, args.newline=TRUE)"
    R -s -e "writeLines(styler::style_text(readLines('stdin'),indent_by = 4L))"
}

function formatXML {
    sed '/./,$!d' | xmllint -
}

function formatSH {
    shfmt - | expand -t 4
}

function formatKEEP {
    cat
}

function format_sel {
    FILE="$1"
    case "$FILE" in
    *.[rR][tT])
        echo formatRT $(format_sel ${FILE%.[Rr][Tt]})
        return 0
        ;;
    *.[rR])
        echo formatR
        return 0
        ;;
    *.sh)
        echo formatSH
        return 0
        ;;
    *.xml)
        echo formatXML
        return 0
        ;;
    *.cpp | *.h | *.hpp | *.cu | *.c | *.cuh)
        echo formatCPP "$FILE"
        return 0
        ;;
    *.*)
        echo formatKEEP
        return 0
        ;;
    esac
    if ! test -f "$1"; then
        echo formatKEEP
        return 0
    fi
    case "$(head -n 1 $1)" in
    *bash)
        echo formatSH
        return 0
        ;;
    *R | *Rscript)
        echo formatR
        return 0
        ;;
    *)
        echo formatKEEP
        return 0
        ;;
    esac
    echo formatKEEP
    return -1
}

function format_to {
    if test -z "$1" || test -z "$2"; then
        echo "Empty arguments to format_to"
        exit 5
    fi
    if ! test -f "$1"; then
        echo "File not exists: $1"
        exit 7
    fi
    F="$(format_sel "$1")"
    if $SKIP_ON_KEEP && test "$F" == "formatKEEP"; then
        $PRINTSKIP && echo "Skipping $1"
        return 0
    fi
    mkdir -p $PP/.format
    NAMESUM=$(echo "$1 $2" | sha256sum | cut -c 1-30)
    SUMFILE="$PP/.format/$NAMESUM.sum"
    if test -f "$SUMFILE"; then
        if sha256sum --status --check "$SUMFILE"; then
            $PRINTSKIP && echo "Skipping $1"
            return 0
        fi
    fi
    ASSUMEFILE="$(basename "$1" | sed 's/[.][Rr][Tt]//')"
    echo "Running: $1 -> $F -> $2"
    cp "$1" tmp
    if cat "$1" | $F >tmp; then
        if test -f "$2"; then
            if diff tmp $2 >/dev/null; then
                rm tmp
            else
                mv tmp "$2"
            fi
        else
            mkdir -p $(dirname "$2")
            mv tmp "$2"
        fi
        sha256sum "$1" "$2" "$FORMATFILE" "$THISSCRIPT" >"$SUMFILE"
    else
        echo "$F failed"
        exit -1
    fi
}

function tmp_before_suffix {
    sed -E 's/(([.][^/.]*)*)$/_format\1/'
}

OUTFILE=""
OUTSAME=false
while test -n "$1"; do
    case "$1" in
    --help)
        echo ""
        echo "$0 [--all] [-o output] [-x] [files]"
        echo ""
        echo "  -o (--output) OUTPUT        : put the formated output in OUTPUT"
        echo "  -x (--overwrite)            : put the formated output in the same file"
        echo "  --all [FROM_DIR] [TO_DIR]   : format all source files in DIR"
        echo ""
        exit 0
        ;;
    -o | --output)
        shift
        OUTFILE="$1"
        ;;
    -x | --overwrite)
        OUTSAME=true
        ;;
    --code)
        DIFFTOOL="code --diff"
        ;;
    --meld)
        DIFFTOOL="meld"
        ;;
    -a | --all)
        shift
        FROM_DIR="src"
        if ! test -z "$1"; then
            if ! test -d "$1"; then
                echo "$1: not a directory"
                exit 3
            fi
            FROM_DIR="$1"
            shift
        fi
        TO_DIR="$FROM_DIR"
        if ! test -z "$1"; then
            if ! test -d "$1"; then
                echo "$1: not a directory"
                exit 3
            fi
            TO_DIR="$1"
            shift
        fi
        if test "$FROM_DIR" == "$TO_DIR" && ! $OUTSAME; then
            echo "Trying to format all file into the same directory. If you want to overwrite say '-x'"
            exit 4
        fi
        PRINTSKIP=false
        SKIP_ON_KEEP=true
        find "$FROM_DIR" -not -path '*/.*' -type f | while read i; do
            if test "$FROM_DIR" == "$TO_DIR"; then
                j="$i"
            else
                j="$TO_DIR/${i#$FROM_DIR}"
            fi
            format_to "$i" "$j"
        done
        ;;
    -p | --pipe)
        formatRT
        ;;
    -*)
        echo "Uknown option $1"
        exit 1
        ;;
    *)
        INFILE="$1"
        if $OUTSAME && test -z "$OUTFILE"; then
            OUTFILE="$INFILE"
        fi
        if ! test -z "$DIFFTOOL"; then
            if test -z "$OUTFILE"; then
                mkdir -p $PP/.format
                OUTFILE="$PP/.format/$(basename "$INFILE")"
            elif test "$OUTFILE" == "$INFILE"; then
                mkdir -p $PP/.format
                INFILE="$PP/.format/$(basename "$INFILE")"
                cp "$OUTFILE" "$INFILE"
            fi
        fi
        if test -z "$OUTFILE"; then
            cat "$INFILE" | $(format_sel "$INFILE")
        else
            format_to "$INFILE" "$OUTFILE"
        fi
        if ! test -z "$DIFFTOOL"; then
            $DIFFTOOL "$OUTFILE" "$INFILE"
        fi
        if test -z "$OUTFILE"; then
            OUTFILE=""
        fi
        ;;
    esac
    shift
done

exit 0
