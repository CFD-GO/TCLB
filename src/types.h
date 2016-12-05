#ifndef TYPES_H
  #define TYPES_H

  #define STRING_LEN 1024
  #ifdef CALC_DOUBLE_PRECISION
    typedef double real_t;
  #else
    typedef float real_t;
  #endif
  typedef struct {real_t x,y,z;} vector_t;
//  typedef struct {real_t x,y,z;} vector_t_b;

//  typedef char flag_t;
  typedef unsigned short int flag_t;

  typedef unsigned char cut_t;
  
  #define NO_CUT 255
  #define CUT_MAX 200
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
