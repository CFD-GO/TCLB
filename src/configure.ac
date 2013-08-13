#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.68])
AC_INIT([CLB], 0.01)
AC_CONFIG_AUX_DIR([tools])
AC_CONFIG_HEADERS([config.h])

AC_ARG_WITH([gcc],
	AC_HELP_STRING([--with-gcc=compiler_bindir],
		[specify the directory containing gcc, e.g. /usr/bin]),
	[COMPILER_BINDIR="$withval"])

AC_ARG_ENABLE([cuda],
	AC_HELP_STRING([--enable-cuda=cuda_home],
		[specify the root of your cuda install, e.g. /usr/local/cuda]))

AC_ARG_WITH([mpi-include],
	AC_HELP_STRING([--with-mpi-include=mpi_include],
		[specify the full path to your mpi headers, e.g. /usr/include/mpi]),
	[MPI_INCLUDE="$withval"])

AC_ARG_WITH([mpi-lib],
	AC_HELP_STRING([--with-mpi-lib=mpi_lib],
		[specify the full path to your mpi shared libraries, e.g. /usr/lib/openmpi/]),
	[MPI_LIBS="$withval"])

AC_ARG_ENABLE([graphics],
	AC_HELP_STRING([--enable-graphics],
		[make a GUI version]))

AC_ARG_ENABLE([double],
	AC_HELP_STRING([--enable-double],
		[make a double precision version]))

AC_ARG_WITH([cuda-arch],
	AC_HELP_STRING([--with-cuda-arch=arch],
		[specify the desired CUDA architecture (sm_10/sm_13/sm_20)]))

AC_ARG_WITH([nlopt],
	AC_HELP_STRING([--with-nlopt=nlopt],
		[specify the full path to your nlopt library]))
		
NVFLAGS=""

# Checks for programs.
AC_PROG_CXX
AC_PROG_CC

# Checks for header files.
AC_CHECK_HEADERS([float.h malloc.h stddef.h stdint.h stdlib.h string.h wchar.h],[],[AC_MSG_ERROR([Cannot find standart headers])])
AC_CHECK_LIB([m], [sqrt],[],[AC_MSG_ERROR([Didn't find math Library])])

if test "x${enable_double}" == "xyes"
then
	AC_DEFINE([CALC_DOUBLE_PRECISION], [1], [Making a double precision version])
fi

if test "x${enable_cuda}" != "xno"
then
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
	AC_CHECK_FILE([${CUDA_HOME}/lib64/libcudart.so], [CUDA_LIB_DIR="/lib64"])
	if test -z "${CUDA_LIB_DIR}"; then
		AC_CHECK_FILE([${CUDA_HOME}/lib/libcudart.so], [CUDA_LIB_DIR="/lib"], [AC_MSG_ERROR([Didn't find libcublas.so])])
	fi
	LDFLAGS="${LDFLAGS} -L${CUDA_HOME}${CUDA_LIB_DIR}"
	CPPFLAGS="${CPPFLAGS} -I${CUDA_HOME}/include"
	AC_CHECK_HEADERS([cuda.h],[],[AC_MSG_ERROR([Didn't find cuda.h])])
	AC_CHECK_LIB([cudart], [cudaMalloc],[],[AC_MSG_ERROR([Didn't find CUDA Run-time Library])])
	if test "x${enable_graphics}" == "xyes"
	then
		AC_CHECK_HEADERS([GL/glut.h],[],[AC_MSG_ERROR([Didn't find GLUT headers])])
		AC_CHECK_LIB([glut], [glutInit],[],[AC_MSG_ERROR([Didn't find GLUT library])])
		AC_DEFINE([GRAPHICS], [1], [Making a graphics version])
	fi

	GRID3D=""
	if test -z "${with_cuda_arch}"; then
		CUDA_ARCH="sm_11"
	else
		CUDA_ARCH="${with_cuda_arch}"
	fi
	AC_MSG_CHECKING([target CUDA architecture])
	if test "$CUDA_ARCH" == "sm_11"
	then
		AC_MSG_RESULT([${CUDA_ARCH}])
		if test "x${enable_double}" == "xyes"
		then
			AC_MSG_ERROR([Double precision not supported in sm_11])
		fi
	elif test "$CUDA_ARCH" == "sm_20"
	then
		AC_MSG_RESULT([${CUDA_ARCH}])
		AC_DEFINE([GRID3D], [1], [Using 3D block grid in CUDA])
	else
		AC_MSG_ERROR([unknown arch ${CUDA_ARCH}])
	fi
	
	if test -z "${CUDA_ARCH}"
	then
		AC_MSG_ERROR([CUDA_ARCH should be set by now])
	else
		AC_DEFINE([CUDA_ARCH], [${CUDA_ARCH}], [target CUDA arch])
		NVFLAGS="${NVFLAGS} -arch ${CUDA_ARCH}"
	fi
else
	AC_DEFINE([CROSS_CPU], [1], [Making a CPU version])
	if test "x${enable_graphics}" == "xyes"
	then
		AC_MSG_ERROR([Cannot make GLUT graphics with CPU version])
	fi
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
AC_CHECK_LIB([mpi], [MPI_Recv],[],[AC_MSG_ERROR([Didn't find MPI Library])])

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

AC_SUBST(NVFLAGS)
AC_CONFIG_FILES([config.mk])
AC_OUTPUT
