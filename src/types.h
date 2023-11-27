#ifndef TYPES_H
#define TYPES_H

#include <limits>

#include "cross.h"

#define STRING_LEN 1024
#ifdef CALC_DOUBLE_PRECISION
typedef double real_t;
#else
typedef float real_t;
#endif
typedef struct {
    real_t x, y, z;
} vector_t;

#ifndef STORAGE_BITS
#ifdef CALC_DOUBLE_PRECISION
typedef double storage_t;
#else
typedef float storage_t;
#endif
#elif STORAGE_BITS == 16
typedef short int storage_t;
#elif STORAGE_BITS == 32
typedef int storage_t;
#elif STORAGE_BITS == 64
typedef long long int storage_t;
#endif

#ifdef STORAGE_SHIFT
CudaDeviceFunction real_t storage_to_real(storage_t);
CudaDeviceFunction storage_t real_to_storage(real_t);
inline CudaDeviceFunction real_t storage_to_real_shift(const storage_t& v, const real_t& shft) {
    return storage_to_real(v) + shft;
}
inline CudaDeviceFunction storage_t real_to_storage_shift(const real_t& v, const real_t& shft) {
    return real_to_storage(v - shft);
}
#else
#define storage_to_real_shift(x__, y__) storage_to_real(x__)
#define real_to_storage_shift(x__, y__) real_to_storage(x__)
#endif
typedef unsigned short int cut_t;

#define NO_CUT 65535
#define CUT_MAX 65000
#define CUT_LEN(x__) (0.005f * (x__))

typedef char tr_flag_t;
typedef int tr_addr_t;
typedef double tr_real_t;

#endif

#ifndef MPI_TYPES_H
#define MPI_TYPES_H
#ifdef CALC_DOUBLE_PRECISION
#define MPI_REAL_T MPI_DOUBLE
#else
#define MPI_REAL_T MPI_FLOAT
#endif
#endif

// #include "TCLBForceGroupCommon.h"
