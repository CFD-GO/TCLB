#include "Consts.h"
#include "types.h"
#include <stdio.h>

#ifndef __CUDACC__
  #ifndef CROSS_HIP
    #define CROSS_CPP
  #endif
#endif

#ifndef CROSS_H

  #ifndef CROSS_CPU
  
    #ifdef CROSS_HIP
     #include <hip/hip_runtime.h>
    #endif
    #ifdef CROSS_CPP
      #ifndef CROSS_HIP
        #include <cuda_runtime.h>
      #endif
      #define CudaDeviceFunction
      #define CudaHostFunction
      #define CudaGlobalFunction
      #define CudaConstantMemory
      template <class T> inline const T& max (const T& x, const T& y) { return x < y ? y : x; };
      template <class T> inline const T& min (const T& x, const T& y) { return x > y ? y : x; };
      inline const real_t max (const real_t& x, const real_t& y) { return x < y ? y : x; };
    #else
//      #include "../../cub/cub/cub.cuh"
      #define CudaDeviceFunction __device__
      #define CudaHostFunction __host__
      #define CudaGlobalFunction __global__
      #define CudaConstantMemory __constant__
      #define CudaSharedMemory __shared__
      #define CudaSyncThreads __syncthreads

      #define CudaSyncThreadsOr(x__) __syncthreads_or(x__)
      #ifdef CROSS_HIP
        #define CudaSyncWarpOr(x__) __any(b);
      #else     
        #if CUDART_VERSION >= 9000
              #define WARP_MASK 0xFFFFFFFF
              #define CudaSyncWarpOr(x__) __any_sync(WARP_MASK, b);
        #elif CUDART_VERSION >= 7000
              #define CudaSyncWarpOr(x__) __any(b);
        #else
              #warning "no atomicSumWarp for this CUDA version"
        #endif
      #endif
      
      #ifndef CROSS_HIP 
       #define CudaKernelRun(a__,b__,c__,d__) a__<<<b__,c__>>>d__; HANDLE_ERROR( cudaDeviceSynchronize()); HANDLE_ERROR( cudaGetLastError() )
       #ifdef CROSS_SYNC
         #define CudaKernelRunNoWait(a__,b__,c__,d__,e__) a__<<<b__,c__>>>d__; HANDLE_ERROR( cudaDeviceSynchronize()); HANDLE_ERROR( cudaGetLastError() );
       #else
         #define CudaKernelRunNoWait(a__,b__,c__,d__,e__) a__<<<b__,c__,0,e__>>>d__;
       #endif
      #else
       #define CudaKernelRun(a__,b__,c__,d__) a__<<<b__,c__>>>d__; HANDLE_ERROR( hipDeviceSynchronize()); HANDLE_ERROR( hipGetLastError() )
       #ifdef CROSS_SYNC
         #define CudaKernelRunNoWait(a__,b__,c__,d__,e__) a__<<<b__,c__>>>d__; HANDLE_ERROR( hipDeviceSynchronize()); HANDLE_ERROR( hipGetLastError() );
       #else
         #define CudaKernelRunNoWait(a__,b__,c__,d__,e__) a__<<<b__,c__,0,e__>>>d__;
       #endif
      #endif
      #define CudaBlock blockIdx
      #define CudaThread threadIdx
      #define CudaNumberOfThreads blockDim
    #endif

   #ifndef CROSS_HIP 
    #define CudaError cudaError_t
    #define CudaSuccess cudaSuccess
    #define CudaGetErrorString(a__) cudaGetErrorString(a__)
    #define CudaExternConstantMemory(x)

    #define CudaMemcpyDeviceToHost cudaMemcpyDeviceToHost
    #define CudaMemcpyHostToDevice cudaMemcpyHostToDevice
    #define CudaCopyToConstant(a__,b__,c__,d__) HANDLE_ERROR( cudaMemcpyToSymbol(b__, c__, d__, 0, cudaMemcpyHostToDevice))
    #define CudaMemcpy2D(a__,b__,c__,d__,e__,f__,g__) HANDLE_ERROR( cudaMemcpy2D(a__, b__, c__, d__, e__, f__, g__) )
    #define CudaMemcpy(a__,b__,c__,d__) HANDLE_ERROR( cudaMemcpy(a__, b__, c__, d__) )
    #ifdef CROSS_SYNC
      #define CudaMemcpyAsync(a__,b__,c__,d__,e__) HANDLE_ERROR( cudaMemcpy(a__, b__, c__, d__) )
      #define CudaMemcpyPeerAsync(a__,b__,c__,d__,e__,f__) HANDLE_ERROR( cudaMemcpyPeer(a__, b__, c__, d__, e__) )
    #else
      #define CudaMemcpyAsync(a__,b__,c__,d__,e__) HANDLE_ERROR( cudaMemcpyAsync(a__, b__, c__, d__, e__) )
      #define CudaMemcpyPeerAsync(a__,b__,c__,d__,e__,f__) HANDLE_ERROR( cudaMemcpyPeerAsync(a__, b__, c__, d__, e__, f__) )
    #endif
    #define CudaMemset(a__,b__,c__) HANDLE_ERROR( cudaMemset(a__, b__, c__) )
    #define CudaMalloc(a__,b__) HANDLE_ERROR( cudaMalloc(a__,b__) )
    #define CudaPreAlloc(a__,b__) HANDLE_ERROR( cudaPreAlloc(a__,b__) )
    #define CudaAllocFinalize() HANDLE_ERROR( cudaAllocFinalize() )
    #define CudaMallocHost(a__,b__) HANDLE_ERROR( cudaMallocHost(a__,b__) )
    #define CudaFree(a__) HANDLE_ERROR( cudaFree(a__) )
    #define CudaFreeHost(a__) HANDLE_ERROR( cudaFreeHost(a__) )
    #define CudaAllocFreeAll() HANDLE_ERROR( cudaAllocFreeAll() )

    #define CudaDeviceCanAccessPeer(a__, b__, c__) HANDLE_ERROR( cudaDeviceCanAccessPeer(a__, b__, c__) )
    #define CudaDeviceEnablePeerAccess(a__, b__) HANDLE_ERROR( cudaDeviceEnablePeerAccess(a__, b__) )

    #define CudaEvent_t cudaEvent_t
    #define CudaEventCreate(a__) HANDLE_ERROR( cudaEventCreate( a__ ) )
    #define CudaEventDestroy(a__) HANDLE_ERROR( cudaEventDestroy( a__ ) )
    #define CudaEventRecord(a__,b__) HANDLE_ERROR( cudaEventRecord( a__, b__ ) )
    #define CudaEventSynchronize(a__) HANDLE_ERROR( cudaEventSynchronize( a__ ) )
    #define CudaEventElapsedTime(a__,b__,c__) HANDLE_ERROR( cudaEventElapsedTime( a__, b__, c__ ) )
    #define CudaStreamWaitEvent(a__,b__,c__) HANDLE_ERROR( cudaStreamWaitEvent( a__, b__, c__) )
    #define CudaDeviceSynchronize() HANDLE_ERROR( cudaDeviceSynchronize() )
    #define CudaEventQuery(a__) cudaEventQuery(a__)

    #define CudaStream_t cudaStream_t
    #define CudaStreamCreate(a__) HANDLE_ERROR( cudaStreamCreate( a__ ) )
    #define CudaStreamSynchronize(a__) HANDLE_ERROR( cudaStreamSynchronize( a__ ) );  HANDLE_ERROR( cudaGetLastError() )

    #define CudaDeviceSynchronize() HANDLE_ERROR( cudaDeviceSynchronize() )

    #define CudaSetDevice(a__) HANDLE_ERROR( cudaSetDevice( a__ ) )
    #define CudaGetDeviceCount(a__) HANDLE_ERROR( cudaGetDeviceCount( a__ ) )
    #define CudaDeviceReset() HANDLE_ERROR( cudaDeviceReset( ) )
    #define CudaFuncAttributes cudaFuncAttributes
    #define CudaFuncGetAttributes(a__,b__) HANDLE_ERROR( cudaFuncGetAttributes(a__, b__) )
   #else
    #define CudaError hipError_t
    #define CudaSuccess hipSuccess
    #define CudaGetErrorString(a__) hipGetErrorString(a__)
    #define CudaExternConstantMemory(x)

    #define CudaMemcpyDeviceToHost hipMemcpyDeviceToHost
    #define CudaMemcpyHostToDevice hipMemcpyHostToDevice
    #define CudaCopyToConstant(a__,b__,c__,d__) HANDLE_ERROR( hipMemcpyToSymbol(b__, c__, d__, 0, hipMemcpyHostToDevice))
    #define CudaMemcpy2D(a__,b__,c__,d__,e__,f__,g__) HANDLE_ERROR( hipMemcpy2D(a__, b__, c__, d__, e__, f__, g__) )
    #define CudaMemcpy(a__,b__,c__,d__) HANDLE_ERROR( hipMemcpy(a__, b__, c__, d__) )
    #ifdef CROSS_SYNC
      #define CudaMemcpyAsync(a__,b__,c__,d__,e__) HANDLE_ERROR( hipMemcpy(a__, b__, c__, d__) )
      #define CudaMemcpyPeerAsync(a__,b__,c__,d__,e__,f__) HANDLE_ERROR( hipMemcpyPeer(a__, b__, c__, d__, e__) )
    #else
      #define CudaMemcpyAsync(a__,b__,c__,d__,e__) HANDLE_ERROR( hipMemcpyAsync(a__, b__, c__, d__, e__) )
      #define CudaMemcpyPeerAsync(a__,b__,c__,d__,e__,f__) HANDLE_ERROR( hipMemcpyPeerAsync(a__, b__, c__, d__, e__, f__) )
    #endif
    #define CudaMemset(a__,b__,c__) HANDLE_ERROR( hipMemset(a__, b__, c__) )
    #define CudaMalloc(a__,b__) HANDLE_ERROR( hipMalloc(a__,b__) )
    #define CudaPreAlloc(a__,b__) HANDLE_ERROR( cudaPreAlloc(a__,b__) )
    #define CudaAllocFinalize() HANDLE_ERROR( cudaAllocFinalize() )
    #define CudaMallocHost(a__,b__) HANDLE_ERROR( hipHostMalloc(a__,b__) )
    #define CudaFree(a__) HANDLE_ERROR( hipFree(a__) )
    #define CudaFreeHost(a__) HANDLE_ERROR( hipHostFree(a__) )
    #define CudaAllocFreeAll() HANDLE_ERROR( cudaAllocFreeAll() )

    #define CudaDeviceCanAccessPeer(a__, b__, c__) HANDLE_ERROR( hipDeviceCanAccessPeer(a__, b__, c__) )
    #define CudaDeviceEnablePeerAccess(a__, b__) HANDLE_ERROR( hipDeviceEnablePeerAccess(a__, b__) )

    #define CudaEvent_t hipEvent_t
    #define CudaEventCreate(a__) HANDLE_ERROR( hipEventCreate( a__ ) )
    #define CudaEventDestroy(a__) HANDLE_ERROR( hipEventDestroy( a__ ) )
    #define CudaEventRecord(a__,b__) HANDLE_ERROR( hipEventRecord( a__, b__ ) )
    #define CudaEventSynchronize(a__) HANDLE_ERROR( hipEventSynchronize( a__ ) )
    #define CudaEventElapsedTime(a__,b__,c__) HANDLE_ERROR( hipEventElapsedTime( a__, b__, c__ ) )
    #define CudaStreamWaitEvent(a__,b__,c__) HANDLE_ERROR( hipStreamWaitEvent( a__, b__, c__) )
    #define CudaDeviceSynchronize() HANDLE_ERROR( hipDeviceSynchronize() )
    #define CudaEventQuery(a__) hipEventQuery(a__)

    #define CudaStream_t hipStream_t
    #define CudaStreamCreate(a__) HANDLE_ERROR( hipStreamCreate( a__ ) )
    #define CudaStreamSynchronize(a__) HANDLE_ERROR( hipStreamSynchronize( a__ ) );  HANDLE_ERROR( hipGetLastError() )

    #define CudaDeviceSynchronize() HANDLE_ERROR( hipDeviceSynchronize() )

    #define CudaSetDevice(a__) HANDLE_ERROR( hipSetDevice( a__ ) )
    #define CudaGetDeviceCount(a__) HANDLE_ERROR( hipGetDeviceCount( a__ ) )
    #define CudaDeviceReset() HANDLE_ERROR( hipDeviceReset( ) )
    #define CudaFuncAttributes hipFuncAttributes
    #define CudaFuncGetAttributes(a__,b__) HANDLE_ERROR( hipFuncGetAttributes(a__, reinterpret_cast<const void*>(&b__)) )
   #endif
//    cudaError_t cudaPreAlloc(void ** ptr, size_t size);
//    cudaError_t cudaAllocFinalize();

    CudaError HandleError( CudaError err, const char *file, int line );
    #define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))
    int GetMaxThreads();
    #define RunKernelMaxThreads (GetMaxThreads())
    #define ISFINITE(l__) isfinite(l__)
  #else
    #include <assert.h>
    #include <time.h>
    #include <cstdlib>
    #include <cstring>
    #include <math.h>
    #ifdef CROSS_OPENMP
      #include <omp.h>
    #endif
    template <class T> inline const T& max (const T& x, const T& y) { return x < y ? y : x; };
    template <class T> inline const T& min (const T& x, const T& y) { return x > y ? y : x; };
    inline const real_t max (const real_t& x, const real_t& y) { return x < y ? y : x; };
    struct float2 { float x,y; };
    struct float3 { float x,y,z; };
    struct double2 { double x,y; };
    struct double3 { double x,y,z; };
    struct uint3 { unsigned int x,y,z; };
    struct dim3 {
      unsigned int x,y,z;
      inline dim3(int x_, int y_, int z_):x(x_),y(y_),z(z_) {};
      inline dim3(int x_, int y_):x(x_),y(y_),z(1) {};
      inline dim3(int x_):x(x_),y(1),z(1) {};
      inline dim3():x(1),y(1),z(1) {};
    };
    struct uchar4 { unsigned char x,y,z,w; };

    struct CudaFuncAttributes_ {
      int 	binaryVersion;
      size_t 	constSizeBytes;
      size_t 	localSizeBytes;
      int 	maxThreadsPerBlock;
      int 	numRegs;
      int 	ptxVersion;
      size_t 	sharedSizeBytes;
    };

    #define CudaError int
    #define CudaSuccess -1
    #define CudaPreAlloc(a__,b__) HANDLE_ERROR( cudaPreAlloc(a__,b__) )
    #define CudaAllocFinalize() HANDLE_ERROR( cudaAllocFinalize() )
    #define CudaAllocFreeAll() HANDLE_ERROR( cudaAllocFreeAll() )

    #define HANDLE_ERROR( err ) (assert( err == CudaSuccess ))


    #define CudaFuncAttributes CudaFuncAttributes_
    #define CudaFuncGetAttributes(a__,b__) {a__->binaryVersion=0; a__->constSizeBytes=-1; a__->localSizeBytes = -1; a__->maxThreadsPerBlock = 32; a__->numRegs = -1;\
      a__->ptxVersion = -1; a__->sharedSizeBytes = -1; }

    #define CudaDeviceFunction
    #define CudaHostFunction
    #define CudaGlobalFunction
    #define CudaConstantMemory
    #define CudaExternConstantMemory(x) extern x
    #define CudaSharedMemory static
    #define CudaSyncThreads() //assert(CpuThread.x == 0)
    #define CudaSyncThreadsOr(x__) x__
    #define CudaSyncWarpOr(x__) x__
    #ifdef CROSS_OPENMP
      #define OMP_PARALLEL_FOR _Pragma("omp parallel for simd")
      #define CudaKernelRun(a__,b__,c__,d__) \
                                      OMP_PARALLEL_FOR \
                                       for (int x__ = 0; x__ < b__.x; x__++) { CpuBlock.x = x__; \
                                        for (CpuBlock.y = 0; CpuBlock.y < b__.y; CpuBlock.y++) \
                                         a__ d__; \
                                       }
    #else
      #define CudaKernelRun(a__,b__,c__,d__) \
                                      for (CpuBlock.y = 0; CpuBlock.y < b__.y; CpuBlock.y++) \
                                       for (CpuBlock.x = 0; CpuBlock.x < b__.x; CpuBlock.x++) \
                                        a__ d__;
    #endif

    #define CudaKernelRunNoWait(a__,b__,c__,d__,e__) CudaKernelRun(a__,b__,c__,d__);
    #define CudaBlock CpuBlock
    #define CudaThread CpuThread
    #define CudaNumberOfThreads CpuSize
    #define CudaCopyToConstant(a__,b__,c__,d__) std::memcpy(&b__, c__, d__)
    #define CudaMemcpy2D(a__,b__,c__,d__,e__,f__,g__) memcpy2D(a__, b__, c__, d__, e__, f__)
    #define CudaMemcpy(a__,b__,c__,d__) memcpy(a__, b__, c__)
    #define CudaMemcpyAsync(a__,b__,c__,d__,e__) CudaMemcpy(a__, b__, c__, d__)
    #define CudaMemset(a__,b__,c__) memset(a__, b__, c__)
    #define CudaMalloc(a__,b__) (assert( (*((void**)(a__)) = malloc(b__)) != NULL ), CudaSuccess)
    #define CudaMallocHost(a__,b__) assert( (*((void**)(a__)) = malloc(b__)) != NULL )
    #define CudaFree(a__) free(a__)
    #define CudaFreeHost(a__) free(a__)


    #define CudaEvent_t double
    #define CudaEventCreate(a__) *a__ = 0
    #define CudaEventDestroy(a__)
    #define CudaEventRecord(a__,b__) a__ = b__
    #ifdef CROSS_OPENMP
      #define CudaEventSynchronize(a__) a__ = omp_get_wtime()*1000
    #else
      #define CudaEventSynchronize(a__) a__ = 1000*((double) clock())/CLOCKS_PER_SEC
    #endif
    #define CudaEventQuery(a__) CudaSuccess
    #define CudaEventElapsedTime(a__,b__,c__) *(a__) = (c__ -  b__)
    #define CudaDeviceSynchronize()
    #define CudaDeviceSynchronize()

    #define CudaStream_t int
    #define CudaStreamCreate(a__)
    #define CudaStreamSynchronize(a__)

    #define CudaDeviceSynchronize()

    #define CudaSetDevice(a__) CpuSize.x=1;CpuSize.y=1;CpuSize.z=1;
    #define CudaGetDeviceCount(a__) *a__ = 1;
    #define CudaDeviceReset()

    #define RunKernelMaxThreads 1
    extern uint3 CpuBlock;
    #ifdef CROSS_OPENMP
      #pragma omp threadprivate(CpuBlock)
    #endif
    extern uint3 CpuThread;
    extern uint3 CpuSize;
    void memcpy2D(void * dst_, int dpitch, void * src_, int spitch, int width, int height);

    template <class T, class P> inline T data_cast(const P& x) { static_assert(sizeof(T)==sizeof(P),"Wrong sizes in data_cast"); T ret; memcpy(&ret, &x, sizeof(T)); return ret; }

    #define __short_as_half(x__)      data_cast<half          , short int     >(x__)
    #define __half_as_short(x__)      data_cast<short int     , half          >(x__)
    #define __int_as_float(x__)       data_cast<float         , int           >(x__)
    #define __float_as_int(x__)       data_cast<int           , float         >(x__)
    #define __longlong_as_double(x__) data_cast<double        , long long int >(x__)
    #define __double_as_longlong(x__) data_cast<long long int , double        >(x__)

//    inline float __int_as_float(int v) { return *reinterpret_cast< float* >(&v); }
//    inline int __float_as_int(float v) { return *reinterpret_cast< int* >(&v); }
//    inline double __longlong_as_double(long long int v) { return *reinterpret_cast< double* >(&v); }
//    inline long long int __double_as_longlong(double v) { return *reinterpret_cast< long long int* >(&v); }

    inline real_t blockSum(real_t val) {
      return val;
    }

    template <typename T>
    inline void atomicSum(T * sum, T val)
    {
      sum[0] += val;
    }

    template <typename T>
    inline void atomicSumWarp(T * sum, T val)
    {
      sum[0] += val;
    }

    template <typename T>
    inline void atomicSumWarpArr(T * sum, T * val, unsigned char len)
    {
      for (unsigned char i = 0; i < len; i ++) sum[i] += val[i];
    }

//    template <typename T>
    inline void atomicSumDiff(real_t * sum, real_t val, bool yes)
      {
        if (yes) sum[0] += val;
      }


    template <typename T>
    inline void atomicMaxReduce(T * sum, T val)
    {
      if (val > sum[0]) sum[0] = val;
    }
  #define ISFINITE(l__) std::isfinite(l__)

  #endif

    CudaError cudaPreAlloc(void ** ptr, size_t size);
    CudaError cudaAllocFinalize();
    CudaError cudaAllocFreeAll();
  
#endif
#define CROSS_H

