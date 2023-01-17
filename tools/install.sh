#!/bin/bash

# --------------- UTILITY FUNCTIONS -------------------------
function usage {
	echo "Usage: Help               : tools/install.sh --help"
	echo "       Install dependency : tools/install.sh [--dry] r|rdep|openmpi|reticulate"
	echo "       Install R package  : tools/install.sh --rpackage [package name or github 'user/repo']"
	exit -2
}

function error {
	echo "$@"
	exit -1
}

function verb {
	if $VERB
	then
		echo "$@"
	fi
}

function rm_tmp {
  if test "x${PWD##*/}" == "xinstall_tmp"
  then
	cd .. && rm -fr install_tmp
  else
  	echo "Not in install_tmp directory while exiting"
  	echo PWD: $PWD
  fi
  return 0;
}
  
function pms_error {
	if test -z "$PMS"
	then
		echo "The package manager needed for installation of '$1'"
	else
		echo "The package manager '$PMS' not supported for installation of '$1'"
	fi
	exit -1;
}

function try {
	comment=$1
	log=$(echo $comment | sed 's|[ /]|.|g').log
	shift
	for RETRY in $(seq $TRYCOUNT)
	do
		FIRST=false; LAST=false; RTXT=""
		test "$RETRY" == 1 && FIRST=true	
		test "$RETRY" == "$TRYCOUNT" && LAST=true
		$FIRST || RTXT=" (retry $RETRY/$TRYCOUNT)"
		if $DRY
		then
			echo "$comment$RTXT:"
			echo "         $@"
			return 0;
		elif $GRPOUTPUT
		then
			echo "::group::$comment$RTXT"
			if "$@"
			then
				echo "::endgroup::"
				return 0;
			else
				echo "---- FAILED ----"
				echo "::endgroup::"
				if $LAST
				then
					exit -1;
				fi
			fi
		else
			echo -n "$comment$RTXT... "
			if "$@" >$log 2>&1
			then
				echo "OK"
				return 0;
			else
				echo "FAILED"
				if $LAST
				then
					echo "----------------- CMD ----------------"
					echo $@
					echo "----------------- LOG ----------------"
					cat  $log
					echo "--------------------------------------"
					exit -1;
				fi
			fi
		fi
		test $TRYDELAY != "0" && sleep $TRYDELAY
	done
	echo "That's weird. We should not arrive here."
	exit -1;
}

function install_rpackage_github {
	name=$(echo $1 | sed "s/\//./g")
	rm -f $name.tar.gz
	try "Downloading $name" wget $WGETOPT https://github.com/$1/archive/master.tar.gz -O $name.tar.gz
	try "Installing $name" R CMD INSTALL $name.tar.gz 
}

function install_rpackage {
	name=$1
	try "Installing $name" R --slave <<EOF
		options(repos='http://cran.rstudio.com');
		p = Sys.getenv("R_LIBS_USER");
		if ( (p != "") && (!file.exists(p)) ) {
			dir.create(p,recursive=TRUE);
			.libPaths(p);
		}
		if (! "$name" %in% available.packages()[,1]) stop("$name not available on CRAN");
		install.packages('$name', method="wget");
		if (! require('$name')) stop("Failed to load $name");
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
		if $DRY
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

# --------------- Move myself to tmp ------------------------
mkdir -p install_tmp >/dev/null 2>&1 || error Failed to create install_tmp directory
cd install_tmp >/dev/null 2>&1 || error Failed to go into install_tmp directory
trap rm_tmp EXIT


DRY=false
GRPOUTPUT=false
TRYCOUNT=1
TRYDELAY=5
WGETOPT=""
PMS=""
GITHUB=false
RSTUDIO_REPO=false
SUDO=""
VERB=false

case "$1" in
-v|--verbose) VERB=true; shift ;;
"") usage ;;
esac

for i in apt-get yum brew
do
	if test -f "$(command -v $i)"
	then 
		verb "Discovered Package Manager: $i"
		PMS=$i
		break
	fi
done

while test -n "$1"
do
	case "$1" in
	--help)
		echo ""
		echo "$0 [-v] [--dry] [--skipssl] ... [things to install]"
		echo ""
		echo "  Things to install:"
		echo "    cuda VERSION   *: Install the nVidia CUDA compilers and libraries"
		echo "    hip  VERSION   *: Install the AMD ROCm/HIP compilers and libraries"
		echo "    openmpi        *: Install the OpenMPI libraries and headers"
		echo "    r              *: Install R Language"
		echo "    rdep            : Install R packages needed by TCLB"
		echo "    essentials     *: Install essential system packages for TCLB (macos)"
		echo ""
		echo "  Other things to install:"
		echo "    rinside         : Install 'rInside' package needed for compiling TCLB with R"
		echo "    python-dev     *: Install Python libraries and headers for compiling TCLB with Python"
		echo "    reticulate      : Install 'reticulate' package needed for using python in rtemplate (<?python ... ?>)"
		echo "    module          : Install 'module' commandline tool"
		echo "    tapenade        : Install TAPENADE for automatic differentiation"
		echo "    lcov            : Install 'lcov' tool for checking code coverage"
		echo ""
		echo "  Install R package:"
		echo "    -r|--rpackage PACKAGE   : Install package from CRAN"
		echo "    -r|--rpackage USER/REPO : Install package from GitHub"
		echo ""
		echo "  Utilities:"
		echo "    submodules      : Clone and/or update git submodules (eg. tests)"
		echo "    gitdep          : Update files pulled from other repositories (according to .gitdep file)"
		echo ""
		echo "  Options:"
		echo "    --dry           : Don't execute anything, just print out"
		echo "    --sudo          : Try using sudo for installation of system packages"
		echo "    --skipssl       : Don't check ssl certs"
		echo "    --pms           : Select Package Menagment System (apt/yum/brew)"
		echo "    -v|--verbose    : Print verbose output"
		echo "    --github        : Prefere github as source of packages"
		echo "    --pms PSM       : select the package manager to use"
		echo "    --rstudio-repo  : use rstudio APT repository for installing R"
		echo "    --retry N       : retry failed steps N times"
		echo "    --retry-delay M : delay M seconds between retries"
		echo "    --group         : Use github actions workflow annotation for output"
		echo "    --help          : display this help message"
		echo ""
		echo "  *) needs sudo"
		echo ""
		exit 0;
		;;
	--dry) DRY=true ;;
	--group) GRPOUTPUT=true ;;
	--retry) shift; TRYCOUNT="$1" ;;
	--retry-delay) shift; TRYDELAY="$1" ;;
	--skipssl) WGETOPT="--no-check-certificate" ;;
	--pms) shift; PMS="$1" ;;
	--github) GITHUB=true ;;
	--rstudio-repo) RSTUDIO_REPO=true ;;
        -v|--verbose) error "-v/--verbose should be the first argument" ;;
	--sudo)
		if test "$UID" == "0"
		then
			verb "--sudo: running as root"
		else
			if test -f "$(command -v sudo)"
			then
				SUDO="sudo -n"
				if $SUDO true 2>/dev/null
				then
					verb "--sudo: sudo working without password"
				else
					error "--sudo: sudo requires a password"
				fi
			else
				error "No sudo command"
			fi
		fi
		;;
	essentials)
		case "$PMS" in
		brew)
			try "Installing gnu utils from brew" brew install coreutils
			try "Installing gnu utils from brew" brew install findutils
			try "Installing gnu utils from brew" brew install gnu-sed
			;;
		esac
		;;
	r)
		case "$PMS" in
		yum)
			try "Adding epel-release repo" $SUDO yum install -y epel-release
			try "Installing R base" $SUDO yum install -y R
			;;
		apt-get)
			if $RSTUDIO_REPO
			then
				CRAN="http://cran.rstudio.com"
				DIST=$(lsb_release -cs)
				if lsb_release -sid | grep "Mint"
				then
					DIST=trusty # All Mints are Trusty :-)
				fi
				try "Adding repository" add-apt-repository "deb ${CRAN}/bin/linux/ubuntu $DIST/"
				try "Adding repository key" apt-key adv --keyserver hkp://keyserver.ubuntu.com:80/ --recv-keys E084DAB9
			fi
			try "Updating APT" $SUDO apt-get update -qq
			try "Installing R base" $SUDO apt-get install -y --allow-unauthenticated --no-install-recommends r-base-dev r-recommended qpdf
			;;
		brew)
			try "Installing R from brew" brew install r
			;;
		*)
			pms_error R ;;
		esac
		#try "Changing access to R lib paths" chmod 2777 /usr/local/lib/R /usr/local/lib/R/site-library
		;;
	-r|--rpackage)
		shift
		test -z "$1" && error "usage tools/install.sh [--github] --rpackage package_name"
		case "$1" in
		*/*) GITHUB=true ;;
		esac
		
		if $GITHUB
		then
			install_rpackage_github "$1"
		else
			install_rpackage "$1"
		fi
		shift
		;;
	rdep)
		if $GITHUB
		then
				install_rpackage_github cran/getopt
				install_rpackage_github cran/optparse
				install_rpackage_github cran/numbers
				install_rpackage_github cran/yaml
		else
				install_rpackage optparse
				install_rpackage numbers
				install_rpackage yaml
		fi
		install_rpackage_github llaniewski/rtemplate
		install_rpackage_github llaniewski/gvector
		install_rpackage_github llaniewski/polyAlgebra
		;;
	rpython)
		install_rpackage rjson
		echo "rPython not supported anymore, installing reticulate instead"
		install_rpackage reticulate
		;;
	reticulate)
		install_rpackage rjson
		install_rpackage reticulate
		;;
	rinside)
		if $GITHUB
		then
			install_rpackage_github eddelbuettel/rinside
		else
			install_rpackage RInside
		fi
		;;
	cuda)
		shift
		test -z "$1" && error "Version number needed for cuda install"
		CUDA=$1
		shift
		echo "#### Installing CUDA library ####"
		
		case "$PMS" in
		apt-get)
			OS=ubuntu1204
			if test "$(lsb_release -si)" == "Ubuntu"
			then
				OS="ubuntu$(lsb_release -sr | sed 's/[.]//g')"
			fi
			KEYRINGVER='1.0-1'
			PINFILE="cuda-${OS}.pin"
			try "Downloading CUDA pin file" wget $WGETOPT http://developer.download.nvidia.com/compute/cuda/repos/${OS}/x86_64/${PINFILE} -O tmp.pinfile
			try "Downloading CUDA keyring file" wget $WGETOPT http://developer.download.nvidia.com/compute/cuda/repos/${OS}/x86_64/cuda-keyring_${KEYRINGVER}_all.deb -O tmp.keyring.deb
			try "Installing CUDA dist" $SUDO dpkg -i tmp.keyring.deb
			try "Planting pin file" $SUDO mv tmp.pinfile /etc/apt/preferences.d/cuda-repository-pin-600
			try "Updating APT" $SUDO apt-get update -qq
			CUDA_APT=${CUDA%-*}
			CUDA_APT=${CUDA_APT/./-}
			try "Installing CUDA form APT" $SUDO apt-get install -y cuda-compiler-${CUDA_APT} cuda-libraries-${CUDA_APT} cuda-libraries-dev-${CUDA_APT}
#			try "Clean APT" $SUDO apt-get clean
			;;
		*)
			pms_error CUDA ;;
		esac
		;;
	hip)
		shift
		test -z "$1" && error "Version number needed for hip install"
		HIP=$1
		shift
		echo "#### Installing HIP library ####"
		IFS=. read V1 V2 V3 <<< $HIP
		echo "Installing version: $HIP ($V1|$V2|$V3)"
		case "$PMS" in
		apt-get)
			OS=xenial
			if test "$(lsb_release -si)" == "Ubuntu"
			then
				OS="$(lsb_release -sc)"
			fi			
			AMDGPU_DEB="$(printf amdgpu-install_%d.%d.%d%02d%02d-1_all.deb "$V1" "$V2" "$V1" "$V2" "$V3")"
			AMDGPU_VER="$HIP"
			try "Updating APT" $SUDO apt-get update
			try "Download AMDGPU install deb" wget https://repo.radeon.com/amdgpu-install/$AMDGPU_VER/ubuntu/$OS/$AMDGPU_DEB
			try "Installing deb" $SUDO apt-get install ./$AMDGPU_DEB
			try "Installing ROCm (amdgpu-install)" $SUDO amdgpu-install -y --usecase=rocm
			;;
		*)
			pms_error HIP ;;
		esac
		;;
	openmpi)
		case "$PMS" in
		yum)
			try "Installing openmpi from yum" $SUDO yum install -y openmpi
			try "Installing openmpi-devel from yum" $SUDO yum install -y openmpi-devel
#			try "Clean yum" $SUDO yum clean packages
			echo "Don't forget to load mpi module before compilation."
			;;
		apt-get)
			try "Updating APT" $SUDO apt-get update -qq
			try "Installing OpenMPI from APT" $SUDO apt-get install -y openmpi-bin libopenmpi-dev
#			try "Clean APT" $SUDO apt-get clean
			;;
		brew)
			try "Installing OpenMPI from brew" brew install openmpi
			;;
		*)
			pms_error OpenMPI ;;
		esac
		;;
	lcov)
		case "$PMS" in
		apt-get)
			try "Installing lcov and time" $SUDO apt-get install -y time lcov
			;;
		*)
			pms_error lcov ;;
		esac
		;;
	submodules)
		if test -f "../tests/README.md"
		then
			echo "\"tests\" already cloned"
			exit 0
		fi
#		try "Saving gitmodules" cp ../.gitmodules gitmodules
#		try "Changing URLs of submodules" sed -i 's/git@github.com:/https:\/\/github.com\//' ../.gitmodules
		try "Updating \"tests\" submodule" git submodule update --init ../tests
#		try "Loading gitmodules" mv gitmodules ../.gitmodules
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
		case "$PMS" in
		yum)
			try "Installing python-devel from yum" $SUDO yum install -y python-devel
			try "Installing numpy from yum" $SUDO yum install -y numpy 
			try "Installing sympy from yum" $SUDO yum install -y sympy
			;;
		apt-get)
			try "Installing python-dev from APT" $SUDO apt-get install -qq python3-dev python3-numpy python3-sympy
			;;
		brew)
			try "Installing Python from brew (this should install headers as well)" brew install python
			;;
		*)
			pms_error python-dev ;;
		esac
		;;
	module)
		case "$PMS" in
		yum)
			try "Installing dependencies: tcl" $SUDO yum -y install tcl 
			try "Installing dependencies: tcl-devel" $SUDO yum -y install tcl-devel
			;;
		*)
			pms_error ;;
		esac
		try "Downloading module" wget https://github.com/cea-hpc/modules/releases/download/v4.1.0/modules-4.1.0.tar.bz2
		try "Unpacking archive" tar -xjf modules-4.1.0.tar.bz2 -C .
		try "Entering module directory" cd modules-4.1.0
		try "Configuring" ./configure
		try "make" make
		try "make install" make install
		try "Leaving module directory" cd ..
		try "Remember to restart terminal" . ~/.bashrc	
		;;
	tapenade)
		if echo "$2" | grep -Eq '^[0-9]*[.][0-9]*$'
		then
			shift
			VER="$1"
		else
			VER="3.16"
		fi
		if test -d ../tapenade
		then
			echo "Looks like tapenade already is installed at '$(cd ../tapenade; pwd))'"
			exit -1
		fi
		try "Downloading Tapenade ($VER)" wget $WGETOPT http://www-sop.inria.fr/ecuador/tapenade/distrib/tapenade_$VER.tar
		try "Unpacking Tapenade" tar xf tapenade_$VER.tar
		if test -d tapenade_$VER
		then
			mv tapenade_$VER ../tapenade
			echo "Installed Tapenade at '$(cd ../tapenade; pwd)'"
		else
			echo "Tapenade installation failed"
		fi
		;;
	-*)
		echo "Unknown option $1" ; usage ;;
	*)		
		echo "Unknown installation '$1'"; usage ;;
	esac
	shift
done

exit 0;
