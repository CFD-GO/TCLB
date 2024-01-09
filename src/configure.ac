#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.


AC_PREREQ([2.63])
AC_INIT([CLB], 0.01)
AC_CONFIG_AUX_DIR([tools])
AC_CONFIG_HEADERS([CLB/config.h:src/config.h.in])
AC_LANG([C++])

AC_ARG_WITH([gcc],
	AS_HELP_STRING([--with-gcc=compiler_bindir],
		[specify the directory containing gcc, e.g. /usr/bin]),
	[COMPILER_BINDIR="$withval"])

AC_ARG_ENABLE([cuda],
	AS_HELP_STRING([--enable-cuda=cuda_home],
		[specify the root of your cuda install, e.g. /usr/local/cuda]))

AC_ARG_ENABLE([hip],
	AS_HELP_STRING([--enable-hip=rocm_home],
		[specify the root of your rocm install, e.g. /opt/rocm]))

AC_ARG_WITH([mpi-include],
	AS_HELP_STRING([--with-mpi-include=mpi_include],
		[specify the full path to your mpi headers, e.g. /usr/include/mpi]),
	[MPI_INCLUDE="$withval"])

AC_ARG_WITH([mpi-lib],
	AS_HELP_STRING([--with-mpi-lib=mpi_lib],
		[specify the full path to your mpi shared libraries, e.g. /usr/lib/openmpi/]),
	[MPI_LIB="$withval"])

AC_ARG_WITH([hdf5],
	AS_HELP_STRING([--with-hdf5=hdf5],
		[specify the full path to your hdf5 installation]))
AC_ARG_WITH([hdf5-lib],
	AS_HELP_STRING([--with-hdf5-lib=hdf5],
		[specify the full path to your hdf5 installation (libraries)]))
AC_ARG_WITH([hdf5-include],
	AS_HELP_STRING([--with-hdf5-include=hdf5],
		[specify the full path to your hdf5 installation (headers)]))

AC_ARG_WITH([parmetis],
	AS_HELP_STRING([--with-parmetis=parmetis],
		[specify the full path to your parmetis installation]))
AC_ARG_WITH([parmetis-lib],
	AS_HELP_STRING([--with-parmetis-lib=parmetis],
		[specify the full path to your parmetis installation (libraries)]))
AC_ARG_WITH([parmetis-include],
	AS_HELP_STRING([--with-parmetis-include=parmetis],
		[specify the full path to your parmetis installation (headers)]))
AC_ARG_WITH([metis],
	AS_HELP_STRING([--with-metis=metis],
		[specify the full path to your metis installation]))
AC_ARG_WITH([metis-lib],
	AS_HELP_STRING([--with-metis-lib=metis],
		[specify the full path to your metis installation (libraries)]))
AC_ARG_WITH([metis-include],
	AS_HELP_STRING([--with-metis-include=metis],
		[specify the full path to your metis installation (headers)]))

AC_ARG_WITH([eigen],
	AS_HELP_STRING([--with-eigen=eigen],
		[specify the full path to your Eigen installation (headers)]))

AC_ARG_ENABLE([graphics],
	AS_HELP_STRING([--enable-graphics],
		[make a GUI version]))

AC_ARG_ENABLE([double],
	AS_HELP_STRING([--enable-double],
		[make a double precision version]))

AC_ARG_WITH([storage],
	AS_HELP_STRING([--with-storage],
		[type of storage used: half, half-shift, float, float-shift, double]))

AC_ARG_ENABLE([cpp17],
	AS_HELP_STRING([--enable-cpp17],
		[enable C++ 17 standard]))

AC_ARG_WITH([cuda-arch],
	AS_HELP_STRING([--with-cuda-arch=arch],
		[specify the desired CUDA architecture (sm_11/sm_13/sm_20/sm_30/sm_60/sm_70/sm_75/sm_80)]))

AC_ARG_WITH([nlopt],
	AS_HELP_STRING([--with-nlopt=nlopt],
		[specify the full path to your nlopt library]))

AC_ARG_WITH([r],
	AS_HELP_STRING([--with-r=r],
		[specify the full path to your r]))

AC_ARG_ENABLE([rinside],
	AS_HELP_STRING([--enable-rinside],
		[specify the full path to your r]))

AC_ARG_WITH([catalyst],
	AS_HELP_STRING([--with-catalyst=path],
		[specify the full path to your paraview source and build]))

AC_ARG_WITH([lammps],
	AS_HELP_STRING([--with-lammps=path],
		[specify the full path to your LAMMPS/LIGGGHTS directory]))

AC_ARG_WITH([lammps-lib],
	AS_HELP_STRING([--with-lammps-lib=libfile],
		[specify the name of the lammps/liggghts library (.so/.a)]))

AC_ARG_WITH([solid_container],
	AS_HELP_STRING([--with-solid-container=type],
		[type of solid (particle) container to use: all/tree/grid]))

AC_ARG_WITH([solid_cache],
	AS_HELP_STRING([--with-solid-cache=N],
		[if to use cache of size N for solid (particle) data]))

AC_ARG_ENABLE([opp],
	AS_HELP_STRING([--enable-opp=N],
		[enable opportunistic operations]))

AC_ARG_WITH([tapenade],
	AS_HELP_STRING([--with-tapenade=path],
		[use tapenade for code differentiation for adjoint]))

AC_ARG_WITH([verbosity_level],
	AS_HELP_STRING([--with-verbosity-level=level],
		[specify level of debug messages 0-all, 1-much, 2-some, 3-normal]))

AC_ARG_ENABLE([debug],
	AS_HELP_STRING([--enable-debug],
		[enables debug version (gcc -g)]))

AC_ARG_ENABLE([waitany],
	AS_HELP_STRING([--enable-waitany],
		[enables MPI WaitAny (default)]))

AC_ARG_WITH([openmp],
	AS_HELP_STRING([--with-openmp],
		[enable openMP in the CPU code]))

AC_ARG_ENABLE([coverage],
	AS_HELP_STRING([--enable-coverage],
		[enable coverage testing]))

AC_ARG_ENABLE([marklines],
	AS_HELP_STRING([--enable-marklines],
		[enable marking lines with pragma in RT]))

AC_ARG_ENABLE([debug-kernel],
	AS_HELP_STRING([--with-debug-kernel],
		[enable NVCC pos compile stats for debug (lots of output!)]))

AC_ARG_WITH([x-block],
	AS_HELP_STRING([--with-x-block],
		[Set the block size in the x direction (X_BLOCK) (default=32)]))
AC_ARG_WITH([x-mod],
	AS_HELP_STRING([--with-x-mod],
		[Set the value used for dividing the memory arrangement in the x direction (default=none)]))
AC_ARG_WITH([warp-size],
	AS_HELP_STRING([--with-warp-size],
		[Set the value used for the warp size]))

AC_ARG_ENABLE([cpu-layout],
	AS_HELP_STRING([--cpu-layout],
		[Enable cpu-optimised memory layout]))


AC_ARG_ENABLE([paranoid],
	AS_HELP_STRING([--enable-paranoid],
		[Makes GCC paranoid.]))
AC_ARG_ENABLE([profiling],
	AS_HELP_STRING([--enable-profiling],
		[enable profiling for R in rtemplate]))		
AC_ARG_ENABLE([keepcode],
	AS_HELP_STRING([--enable-keepcode],
		[keeps the R code for rtemplate along with the generated file]))

AC_ARG_WITH([cpp-flags],
	AS_HELP_STRING([--with-cpp-flags=FLAGS],
		[privide additionals flags for the compiler]),
	[CONF_CPPFLAGS="$withval"])		

if test "x${COMPILER_BINDIR}" != "x"
then
	NVFLAGS="${NVFLAGS} -ccbin=${COMPILER_BINDIR} "
fi

if test -z "$CXX"
then
	if command -v "mpic++"
	then
		CXX="mpic++"
	elif command -v "g++"
	then
		CXX="g++"
	else
		AC_MSG_ERROR([neither mpic++ or g++ found. You can specify compiler with CXX=...])
	fi
fi

mpishow=$($CXX -showme 2>/dev/null)
if ! test -z "$mpishow"
then
	CPPFLAGS="${CPPFLAGS} $(echo $(echo "$mpishow" | sed -e 's| |\n|g' | grep "^-I"))"
	LDFLAGS="${LDFLAGS} $(echo $(echo "$mpishow" | sed -e 's| |\n|g' | grep "^-[[Ll]]"))"
	CXX="$(echo $(echo "$mpishow" | sed -e 's| |\n|g' | sed -n -e '1p'))"
	echo "from mpi showme: CPPFLAGS = $CPPFLAGS"
	echo "from mpi showme: LDFLAGS  = $LDFLAGS"
	echo "from mpi showme: CXX      = $CXX"
fi

# Checks for programs.
AC_PROG_CXX
# AC_PROG_CC

# Checks for header files.
AC_CHECK_HEADERS([float.h stddef.h stdint.h stdlib.h string.h wchar.h],[],[AC_MSG_ERROR([Cannot find standart headers])])
AC_CHECK_LIB([m], [sqrt],[],[AC_MSG_ERROR([Didn't find math Library])])

AC_CHECK_HEADERS([cxxabi.h],[AC_DEFINE([HAS_CXXABI_H], [1], [Has demangle])],[AC_MSG_RESULT([Didn't find cxxabi.h])])


if test "x${enable_double}" != "xno"
then
	AC_DEFINE([CALC_DOUBLE_PRECISION], [1], [Making a double precision version])
fi

if test "x${with_storage}" == "xhalf-shift"
then
	AC_DEFINE([STORAGE_BITS], [16], [Using half as storage])
	AC_DEFINE([STORAGE_SHIFT], [1], [Using shift for storage])
elif test "x${with_storage}" == "xhalf"
then
	AC_DEFINE([STORAGE_BITS], [16], [Using half as storage])
elif test "x${with_storage}" == "xfloat-shift"
then
	AC_DEFINE([STORAGE_BITS], [32], [Using half as storage])
	AC_DEFINE([STORAGE_SHIFT], [1], [Using shift for storage])
elif test "x${with_storage}" == "xfloat"
then
	AC_DEFINE([STORAGE_BITS], [32], [Using half as storage])
elif test "x${with_storage}" == "xdouble"
then
	AC_DEFINE([STORAGE_BITS], [64], [Using half as storage])
elif test "x${with_storage}" != "x"
then
	AC_MSG_ERROR([Not suppported type of storage: x${with_storage}])
fi

OPENMP="${with_openmp}"
test "x${with_openmp}" == "xno" && OPENMP=""
test "x${with_openmp}" == "x" && OPENMP=""
test "x${with_openmp}" == "xyes" && OPENMP="-fopenmp"	

if test "x${enable_hip}" == "xyes"
then
	AC_DEFINE([CROSS_HIP], [1], [Making a HIP version])
	CROSS_HIP=1
	if test "x${enable_graphics}" == "xyes"
	then
		AC_MSG_ERROR([Cannot make GLUT graphics with HIP version])
	fi
	AC_DEFINE([GRID3D], [1], [Using 3D block grid in HIP])
	if test "x${with_x_block}" != "x"
	then
		AC_DEFINE_UNQUOTED([X_BLOCK], ${with_x_block}, [Using X block from --with-x-block])
	else
		AC_DEFINE([X_BLOCK], [32], [Using standard X block])
	fi
	if test "x${with_warp_size}" != "x"
	then
		AC_DEFINE_UNQUOTED([WARPSIZE], ${with_warp_size}, [Using warp size from --with-warp-size])
	else
		AC_DEFINE([WARPSIZE], [32], [Using 32 warpsize to have warp primitives work with x block])
	fi
elif test "x${enable_cuda}" != "xno"
then

    if test "x{debug-kernel}" == "xyes"
    then
     AC_MSG_RESULT(>>>> NVCC debug output enabled <<<<)
	 NVFLAGS="${NVFLAGS} -keep -v"
    fi

	if test "x${enable_cuda}" != "xyes" && test "x${enable_cuda}" != "x"
	then
		CUDA_HOME="${enable_cuda}"
	fi	
	AC_MSG_CHECKING([CUDA home directory])
	if test -z "${CUDA_HOME}"; then
		AC_CHECK_PROG(NVCC,[nvcc],[yes],[no])
		AS_IF(test "x${NVCC}" == "xno",[AC_MSG_ERROR([Didn't find nvcc. You can disable CUDA by --disable-cuda])]) 
		TMP=$(dirname $(which nvcc))
		if test -z "${TMP}"; then
			AC_MSG_RESULT([using default /usr/local/cuda])
			CUDA_HOME="/usr/local/cuda"
		else
			CUDA_HOME="${TMP%/bin}"
			AC_MSG_RESULT([using ${CUDA_HOME}])
		fi	
	else
		AC_MSG_RESULT([using ${CUDA_HOME}])
	fi

	AC_CHECK_FILE([${CUDA_HOME}/bin/nvcc], [NVCC="${CUDA_HOME}/bin/nvcc"],[AC_MSG_ERROR([Cannot find nvcc; Check CUDA install])])

	CUDA_LIB_DIR=""
	AC_CHECK_FILE([${CUDA_HOME}/lib64/libcudart.so], [CUDA_LIB_DIR="${CUDA_HOME}/lib64"])

	
	if test -z "${CUDA_LIB_DIR}"; then
		AC_CHECK_FILE([${CUDA_HOME}/lib/libcudart.so], [CUDA_LIB_DIR="${CUDA_HOME}/lib"], [CUDA_LIB_DIR=""])
	fi
	
	AC_MSG_RESULT([using CUDALIBDIR:${CUDA_LIB_DIR}])
	
	
	if test "x${CUDA_LIB_DIR}" != "x"
	then
		LDFLAGS="${LDFLAGS} -L${CUDA_LIB_DIR}"
		CPPFLAGS="${CPPFLAGS} -I${CUDA_HOME}/include"
	fi
	AC_CHECK_HEADERS([cuda.h],[],[AC_MSG_ERROR([Didn't find cuda.h])])
	AC_CHECK_HEADERS([cuda_runtime.h],[],[AC_MSG_ERROR([Didn't find cuda.h])])
	#AC_CHECK_LIB([cudart], [cudaMalloc],[],[AC_MSG_ERROR([Didn't find CUDA Run-time Library])])
	LDFLAGS="${LDFLAGS} -lcudart"

	if test "x${enable_graphics}" == "xyes"
	then
		AC_CHECK_HEADERS([GL/glut.h],[],[AC_MSG_ERROR([Didn't find GLUT headers])])
		AC_CHECK_LIB([glut], [glutInit],[],[AC_MSG_ERROR([Didn't find GLUT library])])
		AC_CHECK_LIB([GL], [glDrawPixels],[],[AC_MSG_ERROR([Didn't find GLUT library])])
		AC_CHECK_LIB([GLU], [gluSphere],[],[AC_MSG_ERROR([Didn't find GLUT library])])
		AC_DEFINE([GRAPHICS], [1], [Making a graphics version])
	fi

	GRID3D=""
	if test -z "${with_cuda_arch}"; then
		AC_CHECK_LIB([cudart], [cudaGetDeviceProperties])
		AC_MSG_CHECKING([Trying to get compute capability from GPU])
		AC_RUN_IFELSE([AC_LANG_PROGRAM([
			#include <stdio.h>
			#include <cuda_runtime.h>
		],[
			cudaDeviceProp prop;
			cudaError_t status;
			status = cudaGetDeviceProperties(&prop,	0);
			if (status) {
				printf("%s\n", cudaGetErrorString( status ));
				return status;
			}
			printf("%d%d\n", prop.major, prop.minor);
			return 0;
		])],[CUDA_CC=$(./conftest$EXEEXT)],[CUDA_CC="default"; AC_MSG_RESULT([Failed to get CC from GPU])])
	else
		CUDA_CC=$(echo ${with_cuda_arch} | sed -E 's/^(sm_|compute_)//')
	fi
	
	AC_DEFINE_UNQUOTED([CUDA_CC], "${CUDA_CC}", [target CUDA compute capability])
	AC_MSG_CHECKING([target CUDA architecture])
	if test "x${CUDA_CC}" != "xdefault"
	then
		AC_MSG_RESULT([sm_${CUDA_CC}])
		AC_MSG_CHECKING([is CUDA_CC numeric])
		AS_IF(test "x$(printf '%2d' ${CUDA_CC})" == "x$CUDA_CC", [AC_MSG_RESULT([yes])], [AC_MSG_ERROR([Wrong CUDA_CC: $CUDA_CC])]) 
		if test "$CUDA_CC" -lt "20"
		then
			if test "x${enable_double}" == "xyes"
			then
				AC_MSG_ERROR([Double precision not supported in sm_11])
			fi
		else
			AC_DEFINE([GRID3D], [1], [Using 3D block grid in CUDA])
		fi

		#NVFLAGS="${NVFLAGS} -gencode arch=compute_${CUDA_CC},code=sm_${CUDA_CC}"
		NVFLAGS="${NVFLAGS} -arch=compute_${CUDA_CC}"
	else
		AC_MSG_RESULT([nvcc default])
		AC_DEFINE([GRID3D], [1], [Using 3D block grid in CUDA])
	fi
			
	if ! test -z "$OPENMP"
	then
		AC_MSG_ERROR([Cannot make OpenMP support with GPU version])
	fi
	if test "x${with_x_block}" != "x"
	then
		AC_DEFINE_UNQUOTED([X_BLOCK], ${with_x_block}, [Using X block from --with-x-block])
	else
		AC_DEFINE([X_BLOCK], [32], [Using standard X block])
	fi
	if test "x${with_warp_size}" != "x"
	then
		AC_DEFINE_UNQUOTED([WARPSIZE], ${with_warp_size}, [Using warp size from --with-warp-size])
	else
		AC_DEFINE([WARPSIZE], [32], [Using the CUDA standard 32 warp size])
	fi
else
	AC_DEFINE([CROSS_CPU], [1], [Making a CPU version])
	CROSS_CPU=1
	if test "x${enable_graphics}" == "xyes"
	then
		AC_MSG_ERROR([Cannot make GLUT graphics with CPU version])
	fi
	if ! test -z "$OPENMP"
	then
		CPPFLAGS="${CPPFLAGS} $OPENMP"
		LDFLAGS="${LDFLAGS} $OPENMP"
		AC_DEFINE([CROSS_OPENMP], [1], [Using OpenMP])
	fi
	if test "x${with_x_block}" != "x"
	then
		if test "x${with_x_block}" != "x1"
		then
			AC_MSG_ERROR([x-block size not supported for CPU])
		fi
	fi
	AC_DEFINE([X_BLOCK], [1], [Using standard X block])
	if test "x${with_warp_size}" != "x"
	then
		if test "x${with_warp_size}" != "x1"
		then
			AC_MSG_ERROR([warp size size not supported for CPU])
		fi
	fi
	AC_DEFINE([WARPSIZE], [1], [Using the CUDA standard 32 warp size])
	AC_DEFINE([GRID3D], [1], [Using 3D block grid in HIP])
fi

if test "x${with_x_mod}" != "x"
then
	X_MOD="${with_x_mod}"
else
	X_MOD="0"
fi

if test "x${enable_opp}" == "xyes"
then
	AC_DEFINE([USE_ADDOPP], [1], [Opportunistic warp level operations])
fi

if test "x${enable_cpu_layout}" == "xyes"
then
	CPU_LAYOUT="TRUE"
else
	CPU_LAYOUT="FALSE"
fi


AC_MSG_CHECKING([MPI include path])
if test -z "${MPI_INCLUDE}"; then
	if test -z "${MPI}"; then
		MPI_INCLUDE="/usr/include/mpi"
	else
		MPI_INCLUDE="${MPI}/include"
	fi
	AC_MSG_RESULT([${MPI_INCLUDE}])
else
	AC_MSG_RESULT([${MPI_INCLUDE}])
fi
CPPFLAGS="${CPPFLAGS} -I${MPI_INCLUDE}"

AC_MSG_CHECKING([MPI library path])
if test -z "${MPI_LIB}"; then
	if test -z "${MPI}"; then
		MPI_LIB="/usr/lib/openmpi/lib"
	else
		MPI_LIB="${MPI}/lib"
	fi
	AC_MSG_RESULT([${MPI_LIB}])
else
	AC_MSG_RESULT([${MPI_LIB}])
fi
LDFLAGS="${LDFLAGS} -L${MPI_LIB}"
AC_CHECK_HEADERS([mpi.h],[],[AC_MSG_ERROR([Didn't find mpi.h])])
AC_CHECK_LIB([mpi], [MPI_Recv],[],[
	AC_CHECK_FUNC([MPI_Recv],[],[
		AC_MSG_ERROR([Didn't find MPI Library])
	])
])


AC_MSG_CHECKING([EIGEN path])

if test "x${with_eigen}" != "xno"; then
	if test "x${with_eigen}" != "x"; then
		EIGEN_INCLUDE="${with_eigen}"
		if test -d "${EIGEN_INCLUDE}"
		then
			CPPFLAGS="${CPPFLAGS} -I${EIGEN_INCLUDE}"
		else
			AC_MSG_ERROR([${EIGEN_INCLUDE}: No such directory])
		fi
	fi

	AC_CHECK_HEADERS([Eigen/Dense],[],[has_hdf5="no"])
	AC_DEFINE([WITH_EIGEN], [1], [Using Eigen])

fi


AC_MSG_CHECKING([HDF5 path])
need_hdf5="no"
has_hdf5="no"

if test "x${with_hdf5}" != "xno"; then
	if test "x${with_hdf5}" != "x"; then
		HDF5_LIB="${with_hdf5}/lib"
		HDF5_INCLUDE="${with_hdf5}/include"
		need_hdf5="yes"
	fi
	if test "x${with_hdf5_lib}" != "x"
	then
		HDF5_LIB="${with_hdf5_lib}"
		need_hdf5="yes"
	fi
	if test "x${with_hdf5_include}" != "x"
	then
		HDF5_INCLUDE="${with_hdf5_include}"
		need_hdf5="yes"
	fi

	if test "x${HDF5_LIB}" != "x"
	then
		if test -d "${HDF5_LIB}"
		then
			LDFLAGS="${LDFLAGS} -L${HDF5_LIB}"
		else
			AC_MSG_ERROR([${HDF5_LIB}: No such directory])
		fi
	fi
	if test "x${HDF5_INCLUDE}" != "x"
	then
		if test -d "${HDF5_INCLUDE}"
		then
			CPPFLAGS="${CPPFLAGS} -I${HDF5_INCLUDE}"
		else
			AC_MSG_ERROR([${HDF5_INCLUDE}: No such directory])
		fi
	fi
	has_hdf5="yes"
	AC_CHECK_HEADERS([hdf5.h],[],[has_hdf5="no"])
	AC_CHECK_LIB([hdf5], [H5Pset_fapl_mpio],[],[has_hdf5="no"])
	if test "x${has_hdf5}" == "xyes"; then
		AC_DEFINE([WITH_HDF5], [1], [Using HDF5])
	else
		if test "x${need_hdf5}" == "xyes"; then
			AC_MSG_ERROR([HDF5 not found])
		fi
	fi
fi

if test "x${with_parmetis}" != "xno"; then
    AC_MSG_CHECKING([METIS path])
    need_parmetis="no"
    has_parmetis="no"
    need_metis="no"
    has_metis="no"

    # First, try to detect metis
    if test "x${with_metis}" != "x"; then
        if test "x${with_metis}" == "xno"; then
            AC_MSG_ERROR([metis was disabled by the user, but it is needed as a dependency for parmetis])
        fi
    	METIS_LIB="${with_metis}/lib"
    	METIS_INCLUDE="${with_metis}/include"
    	need_metis="yes"
    fi
    if test "x${with_metis_lib}" != "x"
    then
    	METIS_LIB="${with_metis_lib}"
    	need_metis="yes"
    fi
    if test "x${with_metis_include}" != "x"
    then
    	METIS_INCLUDE="${with_metis_include}"
    	need_metis="yes"
    fi
    if test "x${METIS_LIB}" != "x"
    then
    	if test -d "${METIS_LIB}"
    	then
    		LDFLAGS="${LDFLAGS} -L${METIS_LIB}"
    	else
    		AC_MSG_ERROR([${METIS_LIB}: No such directory])
    	fi
    fi
    if test "x${METIS_INCLUDE}" != "x"
    then
    	if test -d "${METIS_INCLUDE}"
    	then
    		CPPFLAGS="${CPPFLAGS} -I${METIS_INCLUDE}"
    	else
    		AC_MSG_ERROR([${METIS_INCLUDE}: No such directory])
    	fi
    fi
    has_metis="yes"
    AC_CHECK_HEADERS([metis.h],[],[has_metis="no"])
    AC_CHECK_LIB([metis],[METIS_PartGraphKway],[],[has_metis="no"])

    # Now, on to parmetis
    AC_MSG_CHECKING([ParMETIS path])
	if test "x${with_parmetis}" != "x"; then
		PARMETIS_LIB="${with_parmetis}/lib"
		PARMETIS_INCLUDE="${with_parmetis}/include"
		need_parmetis="yes"
	fi
	if test "x${with_parmetis_lib}" != "x"
	then
		PARMETIS_LIB="${with_parmetis_lib}"
		need_parmetis="yes"
	fi
	if test "x${with_parmetis_include}" != "x"
	then
		PARMETIS_INCLUDE="${with_parmetis_include}"
		need_parmetis="yes"
	fi

    # Early error if ParMETIS is needed, but METIS (its dependency) is not available
	if test "x${need_parmetis}" == "xyes"; then
	    if test "x${has_metis}" != "xyes"; then
	        AC_MSG_ERROR([parmetis was requested, but metis cannot be compiled and linked correctly])
	    fi
	fi

	if test "x${PARMETIS_LIB}" != "x"
	then
		if test -d "${PARMETIS_LIB}"
		then
			LDFLAGS="${LDFLAGS} -L${PARMETIS_LIB}"
		else
			AC_MSG_ERROR([${PARMETIS_LIB}: No such directory])
		fi
	fi
	if test "x${PARMETIS_INCLUDE}" != "x"
	then
		if test -d "${PARMETIS_INCLUDE}"
		then
			CPPFLAGS="${CPPFLAGS} -I${PARMETIS_INCLUDE}"
		else
			AC_MSG_ERROR([${PARMETIS_INCLUDE}: No such directory])
		fi
	fi
	has_parmetis="yes"
	AC_CHECK_HEADERS([parmetis.h],[],[has_parmetis="no"])
	AC_CHECK_LIB([parmetis],[ParMETIS_V3_PartGeomKway],[],[has_parmetis="no"],[-lmetis])

	if test "x${has_parmetis}" == "xyes"; then
		AC_DEFINE([WITH_PARMETIS], [1], [Using ParMETIS])
	else
		if test "x${need_parmetis}" == "xyes"; then
			AC_MSG_ERROR([parmetis not found])
		fi
	fi
fi

NLOPT=""

AS_CASE([x${with_nlopt}],[xyes],[want_nlopt="yes"],[xno],[want_nlopt="no"],[x],[want_nlopt="maybe"],[want_nlopt="yes"
NLOPT="${with_nlopt}"])

if test "x${want_nlopt}" != "xno"
then
	if test -z "${NLOPT}"
	then
		AC_MSG_CHECKING([for local NLOpt installation])
		if test -d "${PWD}/nlopt"
		then
			AC_MSG_RESULT([found local nlopt directory])
			NLOPT="${PWD}/nlopt"
			local_nlopt="yes"
		else
			AC_MSG_RESULT([not found])		
		fi
	fi

	SV_LDFLAGS="${LDFLAGS}"
	SV_CPPFLAGS="${CPPFLAGS}"
	
	if ! test -z "${NLOPT}"
	then
		LDFLAGS="${LDFLAGS} -L${NLOPT}/lib"
		CPPFLAGS="${CPPFLAGS} -I${NLOPT}/include"
	fi
	nlopt_found="yes"
	AC_CHECK_HEADERS([nlopt.h],[],[nlopt_found="no"])
	AC_CHECK_LIB([nlopt],[nlopt_create],[],[nlopt_found="no"])
	if test "x${nlopt_found}" == "xno"
	then
		if test "x${want_nlopt}" == "xyes"
		then
			AC_MSG_ERROR([Didn't find nlopt libs/headers, but nlopt support requested])
		else
			LDFLAGS="${SV_LDFLAGS}"
		        CPPFLAGS="${SV_CPPFLAGS}"
		fi
	else
		AC_DEFINE([WITH_NLOPT], [1], [Using NLOpt])
	fi
fi

if test "x${with_catalyst}" != "x"
then
	SV_LDFLAGS="${LDFLAGS}"
	SV_CPPFLAGS="${CPPFLAGS}"
	
	if ! test -d "${with_catalyst}"
	then
		AC_MSG_ERROR([Catalyst path ${with_catalyst} is not a directory])
	fi
	
	AC_MSG_CHECKING([for Catalyst-enabled ParaView build directory])
	RET=$(find ${with_catalyst} -name "libvtkPVCatalyst-*.so" -print -quit 2>/dev/null)
	if test -z "${RET}"
	then
		AC_MSG_RESULT([not found])		
		AC_MSG_ERROR([Didn't find Catalyst libs in path ${with_catalyst}])
	fi
	PV_VERSION=$(echo ${RET} | sed 's/^.*libvtkPVCatalyst-\(.*\).so$/\1/')
	RET=$(AS_DIRNAME([${RET}]))
	AC_MSG_RESULT([${RET}])		
	if ! test -d ${RET}
	then
		AC_MSG_ERROR([Lib directory is not a directory - find the developer of this package and discreetly pass him this codeword: Arszenik])
	fi
	PV_BUILD=${RET}
	AC_MSG_CHECKING([for Catalyst-enabled ParaView version])
	if test -z "${PV_VERSION}"
	then
		AC_MSG_RESULT([not found])
		AC_MSG_ERROR([Library didn't have version suffix in ${with_catalyst} (ask the developer, what the hell I'm talking about)])
	fi
	AC_MSG_RESULT([${PV_VERSION}])

	AC_MSG_CHECKING([for Catalyst-enabled ParaView build include directory])
	RET=$(find ${with_catalyst} -name "vtkCommonDataModelModule.h" -print -quit 2>/dev/null)
	if test -z "${RET}"
	then
		AC_MSG_RESULT([not found])		
		AC_MSG_ERROR([Didn't find VTK includes in path ${with_catalyst}])
	fi
	RET=$(AS_DIRNAME([${RET}]))
	RET=$(echo ${RET} | sed 's/\/VTK\/Common\/DataModel$//')
	AC_MSG_RESULT([${RET}])		
	if ! test -d ${RET}
	then
		AC_MSG_ERROR([include directory is not a directory - find the developer of this package and discreetly pass him this codeword: Arszenik])
	fi
	PV_BUILD_INC=${RET}

	AC_MSG_CHECKING([for Catalyst-enabled ParaView source directory])
	RET=$(find ${with_catalyst} -name "vtkCPProcessor.h" -print -quit 2>/dev/null)
	if test -z "${RET}"
	then
		AC_MSG_RESULT([not found])		
		AC_MSG_ERROR([Didn't find Catalyst headers in path ${with_catalyst}])
	fi
	RET=$(AS_DIRNAME([${RET}]))
	RET=$(echo ${RET} | sed 's/\/CoProcessing\/Catalyst$//')
	AC_MSG_RESULT([${RET}])		
	if ! test -d ${RET}
	then
		AC_MSG_ERROR([Header directory is not a directory - find the developer of this package and discreetly pass him this codeword: Arszenik])
	fi
	PV_SOURCE=${RET}
	
	WITH_CATALYST=1
	
fi

if test "x${with_lammps}" != "x"
then
	if ! test -d "${with_lammps}"
	then
		AC_MSG_ERROR([lammps path ${with_lammps} is not a directory])
	fi
	LAMMPS="${with_lammps}"
	LAMMPS="$(cd $LAMMPS; pwd)"
	mkdir -p $LAMMPS/install/include/lammps
	mkdir -p $LAMMPS/install/lib/
	for i in lammps.h library.h input.h modify.h fix.h fix_external.h pointers.h lmptype.h lmpwindows.h atom.h domain.h
	do
		rm ${LAMMPS}/install/include/lammps/${i} 2>/dev/null
		AC_CHECK_FILE([${LAMMPS}/src/${i}], [ln -s ../../../src/${i} ${LAMMPS}/install/include/lammps/${i}],[AC_MSG_ERROR([Cannot find ${i}])])
	done
	for i in partitioner.h
	do
		rm ${LAMMPS}/install/include/lammps/${i} 2>/dev/null
		AC_CHECK_FILE([${LAMMPS}/src/${i}], [ln -s ../../../src/${i} ${LAMMPS}/install/include/lammps/${i}],[])
	done

	for i in error.h comm.h vector_liggghts.h neighbor.h math_extra_liggghts.h domain_I.h
	do
		touch ${LAMMPS}/install/include/lammps/${i}
	done
	if test "x${with_lammps_lib}" == "x"
	then
		for LAMMPS_LIB_DIR in build src
		do
			for LAMMPS_LIB in libliggghts.so liblammps.so
			do
				if test -f "${LAMMPS}/${LAMMPS_LIB_DIR}/${LAMMPS_LIB}"
				then
					break 2
				fi
			done
		done
	else
		LAMMPS_LIB="${with_lammps_lib}"
	fi
	AC_CHECK_FILE([${LAMMPS}/$LAMMPS_LIB_DIR/$LAMMPS_LIB], [ln -s ../../$LAMMPS_LIB_DIR/$LAMMPS_LIB ${LAMMPS}/install/lib/$LAMMPS_LIB],[AC_MSG_ERROR([Cannot find $LAMMPS_LIB])])
	LDFLAGS="${LDFLAGS} -L${LAMMPS}/install/lib -Wl,-rpath=${LAMMPS}/install/lib"
	CPPFLAGS="${CPPFLAGS} -I${LAMMPS}/install/include"
	LAMMPS_LIB_NAME=${LAMMPS_LIB#lib}
	LAMMPS_LIB_NAME=${LAMMPS_LIB_NAME%.*}
	AC_CHECK_HEADERS([lammps/library.h])
	AC_CHECK_LIB([$LAMMPS_LIB_NAME],[lammps_open], [], [AC_MSG_ERROR([Cannot link with $LAMMPS_LIB_NAME])])
	WITH_LAMMPS=1
fi

if test "x${with_solid_container}" == "x"
then
	with_solid_container="grid"
fi

if test "x${with_solid_container}" == "xall"
then
	AC_DEFINE([SOLID_CONTAINER], [1], [Using all container])
elif test "x${with_solid_container}" == "xtree"
then
	AC_DEFINE([SOLID_CONTAINER], [2], [Using tree container])
elif test "x${with_solid_container}" == "xgrid"
then
	AC_DEFINE([SOLID_CONTAINER], [3], [Using grid container])
else
	AC_MSG_ERROR([Not suppported type of storage: x${with_storage}])
fi

if test "x${with_solid_cache}" != "x"
then
	AC_DEFINE_UNQUOTED([SOLID_CACHE], ${with_solid_cache}, [Using solid cache])
fi


if test "x${with_r}" == "x"
then
	AC_PATH_PROG(with_r,[R],[no])
fi

if test "x${with_r}" == "xno"
then
	AC_MSG_ERROR([Cannot run without R (--without-r not allowed)])
else
	R_BIN=${with_r}
fi

AC_MSG_CHECKING($R_BIN)
if ! test -f "$R_BIN"
then
	AC_MSG_ERROR([not found])
else
	AC_MSG_RESULT([exists])
fi

AC_MSG_CHECKING([R package: optparse])
AS_IF([$R_BIN --vanilla --slave -e 'if (!require("optparse")) stop("No");' >/dev/null 2>&1],AC_MSG_RESULT([yes]),AC_MSG_ERROR([no]))
#AC_MSG_CHECKING([R package: numbers])
#AS_IF([$R_BIN --vanilla --slave -e 'if (!require("numbers")) stop("No");' >/dev/null 2>&1],AC_MSG_RESULT([yes]),AC_MSG_ERROR([no]))
AC_MSG_CHECKING([R package: rtemplate])
AS_IF([$R_BIN --vanilla --slave -e 'if (!require("rtemplate")) stop("No");' >/dev/null 2>&1],AC_MSG_RESULT([yes]),AC_MSG_ERROR([no]))
AC_MSG_CHECKING([R package: gvector])
AS_IF([$R_BIN --vanilla --slave -e 'if (!require("gvector")) stop("No");' >/dev/null 2>&1],AC_MSG_RESULT([yes]),AC_MSG_ERROR([no]))
AC_MSG_CHECKING([R package: polyAlgebra])
AS_IF([$R_BIN --vanilla --slave -e 'if (!require("polyAlgebra")) stop("No");' >/dev/null 2>&1],AC_MSG_RESULT([yes]),AC_MSG_ERROR([no]))

if test "x${enable_rinside}" != "xno"
then
	AC_MSG_CHECKING([R package: RInside])
	RINSIDE=yes
	AS_IF([$R_BIN --vanilla --slave -e 'if (!require("RInside")) stop("No");' >/dev/null 2>&1],AC_MSG_RESULT([yes]),[AC_MSG_RESULT([no]); RINSIDE=no])
	if test "x${RINSIDE}" == "xno"
	then
		AC_MSG_CHECKING([if RInside required])
		if test "x${enable_rinside}" == "xyes"
		then
			AC_MSG_ERROR([yes])
		else
			AC_MSG_RESULT([no])
		fi
	else
		function clean_options {
			sed 's/"//g'   # For some reason Rcpp prints the "-I" option with quotations
		}
		CPPFLAGS="${CPPFLAGS} $($R_BIN CMD config --cppflags | clean_options)"
		CPPFLAGS="${CPPFLAGS} $($R_BIN --vanilla --slave -e 'RInside:::CxxFlags()' | clean_options)"
		CPPFLAGS="${CPPFLAGS} $($R_BIN --vanilla --slave -e 'Rcpp:::CxxFlags()' | clean_options)"
		LDFLAGS="${LDFLAGS} $($R_BIN CMD config --ldflags | clean_options)"
		LDFLAGS="${LDFLAGS} $($R_BIN CMD config BLAS_LIBS | clean_options)"
		LDFLAGS="${LDFLAGS} $($R_BIN CMD config LAPACK_LIBS | clean_options)"
		LDFLAGS="${LDFLAGS} $($R_BIN --vanilla --slave -e 'Rcpp:::LdFlags()' | clean_options)"
		LDFLAGS="${LDFLAGS} $($R_BIN --vanilla --slave -e 'RInside:::LdFlags()' | clean_options)"
		AC_CHECK_HEADERS([RInside.h],[],AC_MSG_ERROR([RInside.h not found. If you don't want RInside go --disable-rinside]))
		AC_DEFINE([WITH_R], [1], [Using R])
	fi
fi

if test "x${with_tapenade}" != "xno"
then
	if test "x${with_tapenade}" == "xyes" || test "x${with_tapenade}" == "x"
	then
		if test -f "tapenade/bin/tapenade"
		then
			TAPENADE="$PWD/tapenade/bin/tapenade"
		else
			TAPENADE="tapenade"
		fi
	else
		if test -d "${with_tapenade}"
		then
			TAPENADE="$(cd ${with_tapenade}; pwd)/bin/tapenade"
		else
			AC_MSG_ERROR([Path '${with_tapenade}' is not a directory])
		fi
	fi
	
	
	AC_MSG_CHECKING([for tapenade])
	if command -v "${TAPENADE}" >/dev/null 2>&1
	then
		AC_MSG_RESULT([yes])
		TAPENADE="$(command -v "${TAPENADE}")"
	else
		AC_MSG_RESULT([no])
		TAPENADE=""
		if test "x${with_tapenade}" != "x"
		then
			AC_MSG_ERROR([Tapenade '$TAPENADE' is not a valid executable])
		fi
	fi
fi
		
		


if test "x${enable_debug}" == "xyes"
then
	if test "x${with_verbosity_level}" == "x"
	then
		with_verbosity_level="0"
	fi
	AC_DEFINE([CROSS_SYNC], [1], [Sync calls])
	AC_DEFINE([CROSS_SYNCALLOC], [1], [Sync malloc calls])
	CPPFLAGS="${CPPFLAGS} -g"
	NVFLAGS="${NVFLAGS} --ptxas-options=-v"
fi


if test "x${enable_cpp17}" != "xno"
then
	CPPFLAGS="${CPPFLAGS} -std=c++17"
	NVFLAGS="${NVFLAGS} -std=c++17"
fi

AC_CHECK_HEADERS([chrono],[AC_DEFINE([USE_STEADY_CLOCK], [1], [Using chrono's steady_clock])],[])


if test "x${CONF_CPPFLAGS}" == "x"
then
        if test "x${enable_debug}" == "xyes"
        then
                CPPFLAGS="${CPPFLAGS} -g"
        else
                CPPFLAGS="${CPPFLAGS} -O3"
        fi
        CPPFLAGS="${CPPFLAGS} -Wno-write-strings"
else
	CPPFLAGS="${CPPFLAGS} ${CONF_CPPFLAGS}"
fi

if test "x${with_verbosity_level}" == "x0"
then
	AC_DEFINE([DEBUG_LEVEL], [0], [print EVERYTHING!])
elif test "x${with_verbosity_level}" == "x1"
then
	AC_DEFINE([DEBUG_LEVEL], [1], [print insane amount of output])
elif test "x${with_verbosity_level}" == "x2"
then
	AC_DEFINE([DEBUG_LEVEL], [2], [print moderatly sane additional information])
elif test "x${with_verbosity_level}" == "x3"
then
	AC_DEFINE([DEBUG_LEVEL], [3], [print normal output])
elif test "x${with_verbosity_level}" == "x4"
then
	AC_DEFINE([DEBUG_LEVEL], [4], [print only notice/warning/error])
elif test "x${with_verbosity_level}" == "x"
then
	AC_DEFINE([DEBUG_LEVEL], [3], [normal output])
elif test "x${with_verbosity_level}" == "xno"
then
	AC_DEFINE([DEBUG_LEVEL], [3], [normal output])
else
	AC_MSG_ERROR([Unknown verbosity level level: ${with_verbosity_level}. Can be: 0,1,2,3,4;])
fi

if test "x${enable_waitany}" != "xno"
then
	AC_DEFINE([CROSS_MPI_WAITANY], [1], [MPI WAIT ANY])
fi

if test "x${enable_coverage}" == "xyes"
then
	CPPFLAGS="${CPPFLAGS} -fprofile-arcs -ftest-coverage"
	LDFLAGS="${LDFLAGS} -fprofile-arcs -ftest-coverage"
fi

CP=cp
if test "x${enable_marklines}" == "xyes"
then
	RTOPT="${RTOPT} --mark-lines -i linemark.R"
	CP=./tools/cp_line
fi

if test "x${enable_paranoid}" == "xyes"
then
	CPPFLAGS="${CPPFLAGS} -Wall -Werror -Wno-unknown-warning-option"
	CPPFLAGS="${CPPFLAGS} -Wno-unused-but-set-variable -Wno-unused-variable -Wno-format-overflow -Wno-unused-private-field -Wno-self-assign -Wno-unknown-pragmas"
fi

if test "x${enable_profiling}" == "xyes"
then
	RTOPT="${RTOPT} -p"
fi

if test "x${enable_keepcode}" == "xyes"
then
	RTOPT="${RTOPT} --keep-code"
fi

# Checks for typedefs, structures, and compiler characteristics.
#AC_HEADER_STDBOOL
#AC_C_INLINE
#AC_TYPE_SIZE_T
#AC_TYPE_UINT16_T
#AC_TYPE_UINT32_T
#AC_TYPE_UINT8_T
#AC_CHECK_TYPES([ptrdiff_t])

# Checks for library functions.
#AC_FUNC_ERROR_AT_LINE
#AC_FUNC_MALLOC
#AC_FUNC_STRTOD
#AC_CHECK_FUNCS([floor memmove memset pow sqrt strchr strstr strtol strtoul])


AC_MSG_RESULT([CXX      : $CXX])
AC_MSG_RESULT([CPPFLAGS : $CPPFLAGS])
AC_MSG_RESULT([LDFLAGS  : $LDFLAGS])
AC_MSG_RESULT([NVFLAGS  : $NVFLAGS])

AC_SUBST(CP)
AC_SUBST(NVFLAGS)
AC_SUBST(CROSS_CPU)
AC_SUBST(PV_BUILD)
AC_SUBST(PV_BUILD_INC)
AC_SUBST(PV_SOURCE)
AC_SUBST(CROSS_CPU)
AC_SUBST(CROSS_HIP)
AC_SUBST(WITH_CATALYST)
AC_SUBST(WITH_LAMMPS)
AC_SUBST(WITH_R)
AC_SUBST(TAPENADE)
AC_SUBST(PV_VERSION)
AC_SUBST(RTOPT)
AC_SUBST(X_BLOCK)
AC_SUBST(WARPSIZE)
AC_SUBST(X_MOD)
AC_SUBST(CPU_LAYOUT)

AC_CONFIG_FILES([CLB/config.mk:src/config.mk.in])
AC_CONFIG_FILES([CLB/config.R_:src/config.R.in])
AC_CONFIG_FILES([config.main.mk:src/config.main.mk.in])
AC_OUTPUT

if test -f "CLB/config.R"
then
	if diff "CLB/config.R" "CLB/config.R_" >/dev/null 2>&1
	then
		echo "CLB/config.R unchanged"
	else
		mv CLB/config.R_ CLB/config.R
	fi
else
	mv CLB/config.R_ CLB/config.R
fi
