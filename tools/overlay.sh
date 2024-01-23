#!/bin/bash

set -e

function parse_url {
    echo ${1} | sed -n -E "s%^(([^:/@.]*://|)([^@/]+@|)([^:/@]+)[:/]([^:/@.][^:@]*)|)(@([^:@.]*)|)\$%\\${2}%p"
}

function check_url {
    if test -z "$2"; then
        echo "no url provided for $1" >&2
        exit -1
    fi
    if test "$2" != "$(parse_url "$2" "0")"; then
        echo "failed to parse url for $1: $2" >&2
        exit -1
    fi
}

function check_match {
    if test -n "$4" && test "$3" != "$4"; then
        echo "$1 $2 is different than the one specified:" >&2
        echo " - $2 in the git repo: $3" >&2
        echo " - $2 wanted: $4" >&2
        exit 1
    fi
}

function finish {
    for i in $EXC_SAVED; do
        if test -f "$EXC_TMP_DIR/$i"; then
            mv "$EXC_TMP_DIR/$i" "$i"
        fi
    done
    if test -d "$GIT_OVER"; then
        if test -d ".git"; then
            mv .git check_me_git
        fi
        mv $GIT_OVER .git
    fi
}

function git_init {
    git -c init.defaultBranch=master init
}

function current_branch {
    $1 branch --show-current 2>/dev/null || true
}

function default_branch {
    $1 rev-parse --abbrev-ref origin/HEAD 2>/dev/null | sed -n -E 's%^origin/(HEAD|)%%p'
}

GIT_TCLB=.tclb/git_tclb
GIT_OVER=.tclb/git_over
GITIGN_OVER=".overlay.gitignore"
GITIGN_COMB=".tclb/gitignore"
EXC_FILES="README.md"
UPDATE_SUBM=false
PRINT_HOW_TO=false
EXC_TMP_DIR=.tclb/tmp
WANT_PULL_TCLB=false
WANT_PULL_OVER=false
SAVE_DEFAULTS=false
CONF_FILE=".overlay.config"
while test -n "$1"; do
    case "$1" in
    --submodules)
        UPDATE_SUBM=true
        ;;
    -o | --over | --overlay | --overlay-remote)
        shift
        check_url "overlay" "$1"
        WANT_BRANCH_OVER="$(parse_url "$1" "7")"
        WANT_URL_OVER="$(parse_url "$1" "1")"
        ;;
    -t | --tclb | --tclb-remote)
        shift
        check_url "tclb" "$1"
        WANT_BRANCH_TCLB="$(parse_url "$1" "7")"
        WANT_URL_TCLB="$(parse_url "$1" "1")"
        ;;
    -p | --pull-tclb)
        WANT_PULL_TCLB=true
        ;;
    -s | --save | --save-defaults)
        SAVE_DEFAULTS=true
        ;;
    esac
    shift
done

if test -f "$CONF_FILE"; then
    source "$CONF_FILE"
fi

mkdir -p .tclb
mkdir -p "$EXC_TMP_DIR"

if git tag >/dev/null 2>&1; then
    if git ls-files | grep src/main.cpp; then
        if test -d "$GIT_TCLB"; then
            echo "$GIT_TCLB: already exists"
            exit 1
        fi
        mv .git $GIT_TCLB
        SAVE_DEFAULTS=true
    else
        if test -d "$GIT_OVER"; then
            echo "$GIT_OVER: already exists"
            exit 1
        fi
        mv .git $GIT_OVER && trap finish EXIT
    fi
fi

if ! test -d "$GIT_OVER"; then
    git_init
    mv .git "$GIT_OVER"
fi
trap finish EXIT
function gitover {
    git --git-dir=$GIT_OVER "$@"
}

if ! test -d "$GIT_TCLB"; then
    git_init
    mv .git "$GIT_TCLB"
fi
function gittclb {
    git --git-dir=$GIT_TCLB "$@"
}

URL_OVER="$(gitover remote get-url origin 2>/dev/null || true)"

if test -z "$URL_OVER"; then
    if test -n "$WANT_URL_OVER"; then
        URL_OVER="$WANT_URL_OVER"
        gitover remote add origin $URL_OVER
        WANT_PULL_OVER=true
    fi
else
    check_match "overlay" "remote" "$URL_OVER" "$WANT_URL_OVER"
fi

if test -n "$URL_OVER"; then
    echo "Fetching overlay origin: $URL_OVER"
    gitover fetch origin
    gitover remote set-head origin -a || true
else
    PRINT_HOW_TO=true
    echo "No origin in overlay"
fi

BRANCH_OVER="$(current_branch gitover)"

if test -z "$(gitover branch 2>/dev/null)"; then
    if test -n "$WANT_BRANCH_OVER"; then
        BRANCH_OVER="$WANT_BRANCH_OVER"
    else
        BRANCH_OVER="$(default_branch gitover)"
        if test -z "$BRANCH_OVER"; then
            BRANCH_OVER="master"
        fi
    fi
    WANT_PULL_OVER=true
else
    check_match "overlay" "branch" "$BRANCH_OVER" "$WANT_BRANCH_OVER"
fi

if $WANT_PULL_OVER; then
    echo "Pulling overlay branch: $BRANCH_OVER"
    for i in $EXC_FILES; do
        test -f "$i" && mv "$i" "$EXC_TMP_DIR"
    done
    gitover pull origin $BRANCH_OVER || true
    for i in $EXC_FILES; do
        test -f "$i" || mv "$EXC_TMP_DIR/$i" "$i"
    done
fi

URL_TCLB="$(gittclb remote get-url origin 2>/dev/null || true)"

if test -z "$URL_TCLB"; then
    if ! test -z "$WANT_URL_TCLB"; then
        TCLB_OVER="$WANT_URL_TCLB"
    else
        if test -z "$TCLB_FORK"; then
            TCLB_FORK="CFD-GO/TCLB"
        fi
        case "$URL_OVER" in
        git@github.com*)
            URL_TCLB="git@github.com:${TCLB_FORK}.git"
            ;;
        *)
            URL_TCLB="https://github.com/${TCLB_FORK}.git"
            ;;
        esac
    fi
    gittclb remote add origin $URL_TCLB
    WANT_PULL_TCLB=true
else
    check_match "tclb" "remote" "$URL_TCLB" "$WANT_URL_TCLB"
fi

echo "Fetching tclb origin: $URL_TCLB"
gittclb fetch origin
gittclb remote set-head origin -a || true

BRANCH_TCLB="$(current_branch gittclb)"

if test -z "$(gittclb branch 2>/dev/null)"; then
    if test -n "$WANT_BRANCH_TCLB"; then
        BRANCH_TCLB="$WANT_BRANCH_TCLB"
    else
        if test -z "$TCLB_BRANCH"; then
            BRANCH_TCLB="$(default_branch gittclb)"
        else
            BRANCH_TCLB="$TCLB_BRANCH"
        fi
        if test -z "$BRANCH_TCLB"; then
            BRANCH_TCLB="master"
        fi
    fi
    WANT_PULL_TCLB=true
else
    check_match "tclb" "branch" "$BRANCH_TCLB" "$WANT_BRANCH_TCLB"
fi

if $WANT_PULL_TCLB; then
    echo "Pulling tclb branch: $BRANCH_TCLB"
    EXC_SAVED=""
    for i in $EXC_FILES; do
        if test -f "$i"; then
            mkdir -p "$EXC_TMP_DIR"
            mv "$i" "$EXC_TMP_DIR"
            EXC_SAVED="$EXC_SAVED $i"
        fi
    done
    gittclb pull origin $BRANCH_TCLB
fi
echo "repos:"
echo "  tclb:"
echo "    remote: $URL_TCLB"
echo "    branch: $BRANCH_TCLB"
echo "  overlay:"
echo "    remote: $URL_OVER"
echo "    branch: $BRANCH_OVER"

for i in $EXC_FILES; do
    gittclb update-index --assume-unchanged "$i"
done

echo "###" >$GITIGN_COMB
echo "### This file is generated by the init.sh script" >>$GITIGN_COMB
echo "###" >>$GITIGN_COMB
if test -f ".gitignore"; then
    echo "" >>$GITIGN_COMB
    echo "## .gitignore" >>$GITIGN_COMB
    cat .gitignore | sed -e '/discard in overlay/,/^#/ s/^/#/' >>$GITIGN_COMB
fi
if test -f "$GITIGN_OVER"; then
    echo "" >>$GITIGN_COMB
    echo "## $GITIGN_OVER" >>$GITIGN_COMB
    cat $GITIGN_OVER >>$GITIGN_COMB
fi
echo "" >>$GITIGN_COMB
echo "## files tracked by the TCLB repo" >>$GITIGN_COMB
comm <(gitover ls-files | sort) <(gittclb ls-files | sort) -13 | sed 's|^|/|' >>$GITIGN_COMB

gitover config --local core.excludesfile "$GITIGN_COMB"
gitover config --local alias.tclb "!git --git-dir=\"$GIT_TCLB\""

if $UPDATE_SUBM; then
    echo "Updating submodules"
    gittclb submodule init
    gittclb submodule update
fi

echo ""
echo "--------------- Overlay ready ---------------"
echo ""
echo "To make git operations on the overlay repo,"
echo "  use the standard 'git ...' command."
echo "To make git operations on the TCLB repo,"
echo "  use the 'git tclb ...'."
if $PRINT_HOW_TO; then
    echo ""
    echo "You can add the url to the overlay repository by:"
    echo "  > git remote add origin git@github.com/user/repo.git"
    echo "  > git pull origin $BRANCH_OVER"
    echo "   or"
    echo "  > $0 --overlay git@github.com/user/repo.git"
fi

if test $(parse_url "$URL_TCLB" "4") == "github.com"; then
    TCLB_FORK="$(parse_url "$URL_TCLB" "5" | sed 's/.git$//')"
    TCLB_BRANCH="$BRANCH_TCLB"
    TMP_CONF="$EXC_TMP_DIR/$CONF_FILE"
    echo "# saved from the checked out TCLB repo:" >$TMP_CONF
    echo "TCLB_FORK='$TCLB_FORK'" >>$TMP_CONF
    echo "TCLB_BRANCH='$TCLB_BRANCH'" >>$TMP_CONF
    if test -f "$CONF_FILE" && diff "$CONF_FILE" "$TMP_CONF" >/dev/null 2>&1; then
        rm "$TMP_CONF"
    else
        if $SAVE_DEFAULTS; then
            mv "$TMP_CONF" "$CONF_FILE"
            echo ""
            echo "Saved defaults to $CONF_FILE:"
            echo "You can commit them to the repository by:"
            echo "  > git add $CONF_FILE"
            echo "  > git commit"
        elif ! test -f "$CONF_FILE" && test "$TCLB_FORK @ $TCLB_BRANCH" != "CFD-GO/TCLB @ master"; then
            echo ""
            echo "You can save this fork ($TCLB_FORK),"
            echo "  and branch ($TCLB_BRANCH) as default,"
            echo "  by using the '--save' option."
        fi
    fi
fi



