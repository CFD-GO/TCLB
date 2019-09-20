#!/bin/bash

set -e

slug="$TRAVIS_REPO_SLUG"
test -z "$slug" && slug="$(git remote get-url origin | sed -E 's|.*github.com[:/](.*)/(.*).git$|\1/\2|g')"
if test -z "$slug"
then
	echo Cannot identify the repo.
	exit -1
fi

branch="$TRAVIS_BRANCH"
test -z "$branch" && branch="$(git symbolic-ref --short HEAD)"
if test -z "$branch"
then
	echo Cannot identify the branch
	exit -1
fi

if test "$slug" == "CFD-GO/TCLB"
then
	doc_branch="$branch"
else
	doc_branch="$slug/$branch"
fi

name=$(git --no-pager show -s --format="%aN")
email=$(git --no-pager show -s --format="%ae")
git --no-pager show -s --format="%s" >.msg
if ! test -f .msg
then
	echo No commit message file
	exit -1
fi

echo "---------------------------------------------"
echo "  Slug       : $slug"
echo "  Branch     : $branch"
echo "  doc-branch : $doc_branch"
echo "  Name       : $name"
echo "  e-mail     : $email"
cat .msg | sed 's/^/  Commit     : /'
echo "---------------------------------------------"

if test -z "$GH_TOKEN"
then
	if test -z "$TRAVIS_BRANCH"
	then
		echo "No token, trying ssh"
		GH="git@github.com:"
	else
		echo "No token, and not on Travis-CI"
		echo "Exiting without error"
		exit 0;
	fi
else
	GH="https://$GH_TOKEN@github.com/"
fi

echo "Clone the docs ..."
git clone ${GH}CFD-GO/TCLB_docs
if test "$doc_branch" != "master"
then
	(cd TCLB_docs; git checkout -B $doc_branch)
fi

rm -r TCLB_docs/docs/models
cp -r wiki/models/ TCLB_docs/docs/models
rm -r TCLB_docs/docs/XML-Reference
cp -r wiki/xml/ TCLB_docs/docs/XML-Reference

pushd TCLB_docs
	git config user.email "$email"
	git config user.name "$name"
	git add -A
	git commit -F ../.msg | head -n 300
	if test "$doc_branch" != "master"
	then
		git push --force --set-upstream origin $doc_branch
	else
		git push
	fi
popd

