#ifndef TYPES_H
  #define TYPES_H

  #define STRING_LEN 1024
  #ifdef CALC_DOUBLE_PRECISION
    typedef double real_t;
  #else
    typedef float real_t;
  #endif
  typedef struct {real_t x,y,z;} vector_t;

  #ifndef STORAGE_TYPE
    #ifdef CALC_DOUBLE_PRECISION
      typedef double storage_t;
    #else
      typedef float storage_t;
    #endif
  #elif STORAGE_TYPE == 16
      typedef short int storage_t;
  #elif STORAGE_TYPE == 32
      typedef int storage_t;
  #elif STORAGE_TYPE == 64
      typedef long long int storage_t;
  #endif
  typedef unsigned short int cut_t;

  #define NO_CUT 65535
  #define CUT_MAX 65000
  #define CUT_LEN(x__) (0.005f * (x__))

/*
  struct vector_t {
    real_t x,y,z;
    template <typename T> CudaDeviceFunction CudaHostFunction inline operator T () {
      T p;
      p.x = x; p.y = y; p.z = z;
      return p;
    }
  };
*/
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
