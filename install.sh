mkdir -p tmp
pushd tmp
rm * 2>/dev/null

function github_install {
	name=$(echo $1 | sed "s/\//./g")-$2
	echo Downloading $name ...
	wget https://github.com/$1/archive/$2.tar.gz -O $name.tar.gz >$name.wget.log 2>&1 || (echo failed; exit -1);
	echo Installing $name ...
	R CMD INSTALL $name.tar.gz >$name.install.log 2>&1|| (echo failed; exit -1);
}

github_install llaniewski/rtemplate v1.0
github_install llaniewski/gvector v1.0
github_install llaniewski/polyAlgebra v1.0

popd
rm -r tmp
