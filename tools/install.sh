#!/bin/bash

# --------------- UTILITY FUNCTIONS -------------------------
function usage {
	echo "install.sh [dry] [skipssl] r|rdep|cuda|submodules|openmpi|coveralls|python-dev|rpython|module [VERSION]"
	exit -2
}

function dry {
	test $RUN == "dry"
}

function error {
	echo $@
	exit -1
}

function rm_tmp {
  if test "x${PWD##*/}" == "xinstall_tmp"
  then
	cd .. && rm -fr install_tmp
  else
  	echo Not in install_tmp directory while exiting
  	echo PWD: $PWD
  fi
  return 0;
}
  
# --------------- First, check if running dry ----------------
if test "x$1" == "xdry"
then
	RUN=dry
	shift
else
	RUN=normal
fi

if test "x$1" == "xskipssl"
then
	WGETOPT="--no-check-certificate"
	shift
fi

# ------------------- Second, check Package Management System - PMS  ----------------------
PMS=""
function get_PMS {
	if test -z "$PMS"
	then
		pms_array=( apt-get yum )
		for i in "${pms_array[@]}"
		do
			if [ -x "$(command -v $i)" ] ; then 
				echo "Discovered Package Manager: $i"
				PMS=$i
			fi
		done
		if test -z "$PMS"
		then
			echo "Unknown type of Package Manager, only apt-get and yum are supported."
			exit 2;
		fi
	fi
}

# --------------- First argument is type of install ---------
test -z "$1" && usage
inst=$1
shift

# --------------- Move myself to tmp ------------------------
mkdir -p install_tmp >/dev/null 2>&1 || error Failed to create install_tmp directory
cd install_tmp >/dev/null 2>&1 || error Failed to go into install_tmp directory
trap rm_tmp EXIT

# --------------- Install functions -------------------------
function try {
	comment=$1
	log=$(echo $comment | sed 's|[ /]|.|g').log
	shift
	if dry
	then
		echo "$comment:"
		echo "         $@"
	else
		echo -n "$comment... "
		if "$@" >$log 2>&1
		then
			echo "OK"
		else
			echo "FAILED"
			echo "----------------- CMD ----------------"
			echo $@
			echo "----------------- LOG ----------------"
			cat  $log
			echo "--------------------------------------"
			exit -1;
		fi
	fi
	return 0;
}

function github_install {
	name=$(echo $1 | sed "s/\//./g")
	rm -f $name.tar.gz
	try "Downloading $name" wget $WGETOPT https://github.com/$1/archive/master.tar.gz -O $name.tar.gz
	try "Installing $name" R CMD INSTALL $name.tar.gz 
}

function normal_install {
	name=$1
	try "Installing $name" R --slave <<EOF
		options(repos='http://cran.rstudio.com');
		p = Sys.getenv("R_LIBS_USER");
		if ( (p != "") && (!file.exists(p)) ) {
			dir.create(p,recursive=TRUE);
			.libPaths(p);
		}
		install.packages('$name');
EOF
}

function gitdep.cp {
	echo -n "Copy $1... "
	if ! test -f "gitdep_repo/$1"
	then
		echo "No such file"
		exit -1;
	fi
	if ! test -d "../$2"
	then
		echo "Targed directory $2 doesn't exist";
		exit -1;
	fi
	if diff gitdep_repo/$1 ../$2 >/dev/null
	then
		echo "Same"
	else
		echo "Changed"
		if dry
		then
			echo "cp \"gitdep_repo/$1\" \"../$2\""
		else
			cp "gitdep_repo/$1" "../$2"
		fi
	fi
	return 0;
}

function gitdep {
	DIR=$1
	shift
	REPO=$1
	shift
	echo "repo: $REPO dir:$DIR files:$@"
	try "Clone $REPO" git clone $REPO gitdep_repo
	for i in "$@"
	do
		gitdep.cp "$i" "$DIR"
	done
	rm -r gitdep_repo
	return 0;
}

# --------------- Main install script -----------------------

dry && echo Running dry install


case "$inst" in
r)
	get_PMS
	CRAN="http://cran.rstudio.com"
	DIST=$(lsb_release -cs)
	if lsb_release -sid | grep "Mint"
	then
		DIST=trusty # All Mints are Trusty :-)
	fi
	if test "x$PMS" == "xyum"
	then
	    try "Adding epel-release repo" yum install -y epel-release
	    try "Installing R base" sudo yum install -y R
	    try "Changing access to R lib paths" chmod 2777 /usr/lib64/R/library /usr/share/R/library
	fi
	
	if test "x$PMS" == "xapt-get"
	then
	    try "Adding repository" add-apt-repository "deb ${CRAN}/bin/linux/ubuntu $DIST/"
	    try "Adding repository key" apt-key adv --keyserver hkp://keyserver.ubuntu.com:80/ --recv-keys E084DAB9
	    try "Updating APT" apt-get update -qq
	    try "Installing R base" apt-get install -y --no-install-recommends r-base-dev r-recommended qpdf
	    try "Changing access to R lib paths" chmod 2777 /usr/local/lib/R /usr/local/lib/R/site-library
	fi
	;;
rdep)
        if test "x$1" == "xgithub"
        then
                github_install cran/getopt
                github_install cran/optparse
                github_install cran/numbers
                github_install cran/yaml
        else
                normal_install optparse
                normal_install numbers
                normal_install yaml
        fi
	github_install llaniewski/rtemplate
	github_install llaniewski/gvector
	github_install llaniewski/polyAlgebra
	;;
rpython)
	normal_install rjson
	normal_install rPython
	;;
rinside)
	if test "x$1" == "xgithub"
	then
		github_install eddelbuettel/rinside
	else
		normal_install RInside
	fi
	;;
cuda)
	test -z "$1" && error Version number needed for cuda install
	get_PMS
	CUDA=$1
	shift
	echo "#### Installing CUDA library ####"
	
	if test "x$PMS" == "xyum"
	then
		echo "The install script doesnt support yum yet, please install CUDA manually."
		exit 1;
	fi
	
	if test "x$PMS" == "xapt-get"
	then
	    try "Downloading CUDA dist" wget $WGETOPT http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1204/x86_64/cuda-repo-ubuntu1204_${CUDA}_amd64.deb
	    try "Installing CUDA dist" dpkg -i cuda-repo-ubuntu1204_${CUDA}_amd64.deb
	    try "Updating APT" apt-get update -qq
	    CUDA_APT=${CUDA%-*}
	    CUDA_APT=${CUDA_APT/./-}
	    try "Installing CUDA form APT" apt-get install -y cuda-drivers cuda-core-${CUDA_APT} cuda-cudart-dev-${CUDA_APT}
	    try "Clean APT" apt-get clean
	fi
	;;
openmpi)
	get_PMS
	if test "x$PMS" == "xyum"
	then
		try "Installing openmpi from yum" yum install -y openmpi
		try "Installing openmpi-devel from yum" yum install -y openmpi-devel
		try "Clean yum" yum clean packages
		echo "Don't forget to load mpi module before compilation."
	fi
	if test "x$PMS" == "xapt-get"
	then
		try "Updating APT" apt-get update -qq
		try "Installing MPI from APT" apt-get install -y openmpi-bin libopenmpi-dev
		try "Clean APT" apt-get clean
	fi
	;;
coveralls)
	get_PMS
	if test "x$PMS" == "xyum"	
	then
		echo "The install script doesnt support yum yet, please install CUDA manually."
	fi 
	
	if test "x$PMS" == "xapt-get"
	then
		try "Installing lcov" apt-get install -y lcov
		try "Installing time" apt-get install -y time
	#	try "Installing coveralls-lcov" gem install coveralls-lcov
	fi
	;;
submodules)
	if test -f "../tests/README.md"
	then
		echo "\"tests\" already cloned"
		exit 0
	fi
	try "Saving gitmodules" cp ../.gitmodules gitmodules
	try "Changing URLs of submodules" sed -i 's/git@github.com:/https:\/\/github.com\//' ../.gitmodules
	try "Updating \"tests\" submodule" git submodule update --init ../tests
	try "Loading gitmodules" mv gitmodules ../.gitmodules
	;;
gitdep)
	if ! test -f "../.gitdeps"
	then
		echo no .gitdeps file
		exit 0;
	fi
	while read line
	do
		gitdep $line
	done <../.gitdeps
	;;
python-dev)
	get_PMS
	if test "x$PMS" == "xyum"	
	then
		try "Installing python-devel from yum" yum install -y python-devel
		try "Installing numpy from yum" yum install -y numpy 
		try "Installing sympy from yum" yum install -y sympy
	fi
	
	if test "x$PMS" == "xapt-get"
	then
		try "Installing python-dev from APT" apt-get install -qq python-dev python-numpy python-sympy
	fi
	;;
module)
	try "Installing dependencies: tcl" yum -y install tcl 
	try "Installing dependencies: tcl-devel" yum -y install tcl-devel
	try "Downloading module" wget https://github.com/cea-hpc/modules/releases/download/v4.1.0/modules-4.1.0.tar.bz2
	try "Unpacking archive" tar -xjf modules-4.1.0.tar.bz2 -C .
	try "Entering module directory" cd modules-4.1.0
	try "Configuring" ./configure
	try "make" make
	try "make install" make install
	try "Leaving module directory" cd ..
	try "Remember to restart terminal" . ~/.bashrc	
	;;
*)
	echo "Unknown type of install $inst"
	usage
	;;
esac


exit 0;
