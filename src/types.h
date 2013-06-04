#ifndef TYPES_H
  #define TYPES_H

  #define STRING_LEN 1024
  #ifdef CALC_DOUBLE_PRECISION
    typedef double type_f;
  #else
    typedef float type_f;
  #endif
  typedef struct {type_f x,y,z;} type_v;

//  typedef char flag_t;
  typedef unsigned short int flag_t;

/*
  struct type_v {
    type_f x,y,z;
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
    #define MPI_TYPE_F MPI_DOUBLE
  #else
    #define MPI_TYPE_F MPI_FLOAT
  #endif
#endif
