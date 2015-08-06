#!/bin/bash

# --------------- UTILITY FUNCTIONS -------------------------
function usage {
	echo "install.sh [dry] rdep|cuda [VERSION]"
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
  
# --------------- First check if running dry ----------------
if test "x$1" == "xdry"
then
	RUN=dry
	shift
else
	RUN=normal
fi

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
	log=$(echo $comment | sed 's/ /./g').log
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
	try "Downloading $name" wget https://github.com/$1/archive/master.tar.gz -O $name.tar.gz
	try "Installing $name" R CMD INSTALL $name.tar.gz
}

function normal_install {
	name=$1
	try "Installing $name" R -e "options(repos='http://cran.rstudio.com');install.packages('$name')"
}


# --------------- Main install script -----------------------

dry && echo Running dry install

case "$inst" in
r)
	CRAN="http://cran.rstudio.com"
	try "Adding repository" add-apt-repository "deb ${CRAN}/bin/linux/ubuntu $(lsb_release -cs)/"
	try "Adding repository key" apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E084DAB9
	try "Updating APT" apt-get update -qq
	try "Installing R base" apt-get install --no-install-recommends r-base-dev r-recommended qpdf
	try "Changing access to R lib paths" chmod 2777 /usr/local/lib/R /usr/local/lib/R/site-library
	;;
rdep)
	normal_install optparse
	github_install llaniewski/rtemplate
	github_install llaniewski/gvector
	github_install llaniewski/polyAlgebra
	;;
cuda)
	test -z "$1" && error Version number needed for cuda install
	CUDA=$1
	shift
	echo "#### Installing CUDA library ####"
	try "Downloading CUDA dist" wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1204/x86_64/cuda-repo-ubuntu1204_${CUDA}_amd64.deb
	try "Installing CUDA dist" dpkg -i cuda-repo-ubuntu1204_${CUDA}_amd64.deb
	try "Updating APT" apt-get update -qq
	CUDA_APT=${CUDA%-*}
	CUDA_APT=${CUDA_APT/./-}
	try "Installing CUDA form APT" apt-get install -y cuda-drivers cuda-core-${CUDA_APT} cuda-cudart-dev-${CUDA_APT}
	try "Clean APT" apt-get clean
	;;
*)
	echo "Unknown type of install $inst"
	usage
	;;
esac


exit 0;
