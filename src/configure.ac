#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.


AC_PREREQ([2.63])
AC_INIT([CLB], 0.01)
AC_CONFIG_AUX_DIR([tools])
AC_CONFIG_HEADERS([CLB/config.h:src/config.h.in])

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
	[MPI_LIB="$withval"])

AC_ARG_ENABLE([graphics],
	AC_HELP_STRING([--enable-graphics],
		[make a GUI version]))

AC_ARG_ENABLE([double],
	AC_HELP_STRING([--enable-double],
		[make a double precision version]))

AC_ARG_WITH([cuda-arch],
	AC_HELP_STRING([--with-cuda-arch=arch],
		[specify the desired CUDA architecture (sm_11/sm_13/sm_20/sm_30)]))

AC_ARG_WITH([nlopt],
	AC_HELP_STRING([--with-nlopt=nlopt],
		[specify the full path to your nlopt library]))

AC_ARG_WITH([catalyst],
	AC_HELP_STRING([--with-catalyst=path],
		[specify the full path to your paraview source and build]))

AC_ARG_WITH([verbosity_level],
	AC_HELP_STRING([--with-verbosity-level=level],
		[specify level of debug messages 0-all, 1-much, 2-some, 3-normal]))

AC_ARG_ENABLE([debug],
	AC_HELP_STRING([--enable-debug],
		[enables debug version (gcc -g)]))

AC_ARG_ENABLE([waitany],
	AC_HELP_STRING([--enable-waitany],
		[enables MPI WaitAny (default)]))

AC_ARG_ENABLE([openmp],
	AC_HELP_STRING([--enable-openmp],
		[enable openMP in the CPU code]))

AC_ARG_ENABLE([coverage],
	AC_HELP_STRING([--enable-coverage],
		[enable coverage testing]))

AC_ARG_ENABLE([marklines],
	AC_HELP_STRING([--enable-marklines],
		[enable marking lines with pragma in RT]))

AC_ARG_ENABLE([debug-kernel],
	AC_HELP_STRING([--with-debug-kernel],
		[enable NVCC pos compile stats for debug (lots of output!)]))

AC_ARG_WITH([python],
	AC_HELP_STRING([--with-python],
		[Enable CallPython handler, requires python-dev]))

AC_ARG_WITH([python_config],
	AC_HELP_STRING([--with-python-config],
		[Set python(2)-config binary to call]))


AC_ARG_ENABLE([paranoid],
	AC_HELP_STRING([--enable-paranoid],
		[Makes GCC paranoid.]))
AC_ARG_ENABLE([profiling],
	AC_HELP_STRING([--enable-profiling],
		[enable profiling for R in rtemplate]))		
NVFLAGS=""

# Checks for programs.
AC_PROG_CXX
AC_PROG_CC

# Checks for header files.
AC_CHECK_HEADERS([float.h malloc.h stddef.h stdint.h stdlib.h string.h wchar.h],[],[AC_MSG_ERROR([Cannot find standart headers])])
AC_CHECK_LIB([m], [sqrt],[],[AC_MSG_ERROR([Didn't find math Library])])

if test "x${enable_double}" != "xno"
then
	AC_DEFINE([CALC_DOUBLE_PRECISION], [1], [Making a double precision version])
fi

if test "x${enable_cuda}" != "xno"
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
	#AC_CHECK_LIB([cudart], [cudaMalloc],[],[AC_MSG_ERROR([Didn't find CUDA Run-time Library])])
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
		CUDA_ARCH="sm_20"
	else
		CUDA_ARCH="${with_cuda_arch}"
	fi
	AC_MSG_CHECKING([target CUDA architecture])
	CUDA_SUFF=""
	if test "$CUDA_ARCH" == "sm_11"
	then
		AC_MSG_RESULT([${CUDA_ARCH}])
		if test "x${enable_double}" == "xyes"
		then
			AC_MSG_ERROR([Double precision not supported in sm_11])
		fi
		CUDA_SUFF="11"
	elif test "$CUDA_ARCH" == "sm_13"
	then
		AC_MSG_RESULT([${CUDA_ARCH}])
		if test "x${enable_double}" == "xyes"
		then
			AC_MSG_ERROR([Double precision not supported in sm_13])
		fi
		CUDA_SUFF="13"
	elif test "$CUDA_ARCH" == "sm_20"
	then
		AC_MSG_RESULT([${CUDA_ARCH}])
		AC_DEFINE([GRID3D], [1], [Using 3D block grid in CUDA])
		CUDA_SUFF="20"
	elif test "$CUDA_ARCH" == "sm_30"
	then
		AC_MSG_RESULT([${CUDA_ARCH}])
		AC_DEFINE([GRID3D], [1], [Using 3D block grid in CUDA])

		CUDA_SUFF="30"
	else
		AC_MSG_ERROR([unknown arch ${CUDA_ARCH}])
	fi
	
	if test -z "${CUDA_ARCH}"
	then
		AC_MSG_ERROR([CUDA_ARCH should be set by now])
	else
		AC_DEFINE_UNQUOTED([CUDA_ARCH], "${CUDA_ARCH}", [target CUDA arch])
		#NVFLAGS="${NVFLAGS} -gencode arch=compute_${CUDA_SUFF},code=sm_${CUDA_SUFF}"
		NVFLAGS="${NVFLAGS} -arch=compute_${CUDA_SUFF}"
	fi
	if test "x${enable_openmp}" == "xyes"
	then
		AC_MSG_ERROR([Cannot make OpenMP support with GPU version])
	fi
else
	AC_DEFINE([CROSS_CPU], [1], [Making a CPU version])
	CROSS_CPU=1
	if test "x${enable_graphics}" == "xyes"
	then
		AC_MSG_ERROR([Cannot make GLUT graphics with CPU version])
	fi
	if test "x${enable_openmp}" == "xyes"
	then
		CPPFLAGS="${CPPFLAGS} -fopenmp"
		LDFLAGS="${CPPFLAGS} -fopenmp"
		AC_DEFINE([CROSS_OPENMP], [1], [Using OpenMP])
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


if  test "x${with_python}" == "xyes"; then
   
    AC_CHECK_PROG(python_config,"${with_python_config}","${with_python_config}",python-config)
    
    AC_MSG_NOTICE([python config: ${python_config}])

    AC_MSG_CHECKING([python embandinng: linker flags])
    PF=$(${python_config} --ldflags)
    LDFLAGS="${LDFLAGS} ${PF}"
    AC_MSG_RESULT([${PF}])
    
    AC_MSG_CHECKING([Python Embandinng: compiler flags])
#    PF=$($python_config --cflags)
    PF=$($python_config --includes)
    CPPFLAGS="${CPPFLAGS} ${PF}"

    PF=$($python_config --prefix)

    LDFLAGS="${LDFLAGS} -L${PF}/lib"
    EMBEDED_PYTHON=1
    AC_MSG_RESULT([${PF}])
	AC_DEFINE([EMBEDED_PYTHON], [1], [Using Python support])
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
else
	CPPFLAGS="${CPPFLAGS} -O3"
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
	CPP_OPT="${CPP_OPT} -fprofile-arcs -ftest-coverage"
	LDFLAGS="${LDFLAGS} -fprofile-arcs -ftest-coverage"
fi

CP=cp
if test "x${enable_marklines}" == "xyes"
then
	RTOPT="${RTOPT} --mark-lines"
	CP=./tools/cp_line
fi

if test "x${enable_paranoid}" == "xyes"
then
	CPP_OPT="${CPP_OPT} -Wall -Werror -Wno-unused-but-set-variable -Wno-unused-variable"
fi

if test "x${enable_profiling}" == "xyes"
then
	RTOPT="${RTOPT} -p"
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

AC_SUBST(CP)
AC_SUBST(NVFLAGS)
AC_SUBST(CROSS_CPU)
AC_SUBST(PV_BUILD)
AC_SUBST(PV_BUILD_INC)
AC_SUBST(PV_SOURCE)
AC_SUBST(CROSS_CPU)
AC_SUBST(WITH_CATALYST)
AC_SUBST(PV_VERSION)
AC_SUBST(RTOPT)
AC_SUBST(CPP_OPT)
AC_SUBST(EMBEDED_PYTHON)

AC_CONFIG_FILES([CLB/config.mk:src/config.mk.in])
AC_CONFIG_FILES([config.main.mk:src/config.main.mk.in])
AC_OUTPUT
