#include "types.h"

#ifndef CROSS_H

  #ifdef CROSS_CPU
    #define CROSS_CPP
  #endif

  #ifdef CROSS_CPP
    #define CudaDeviceFunction
    #define CudaHostFunction
    #define CudaGlobalFunction
    template <class T> inline const T& max (const T& x, const T& y) { return x < y ? y : x; };
    template <class T> inline const T& min (const T& x, const T& y) { return x > y ? y : x; };
  #else
    #define CudaDeviceFunction __device__
    #define CudaHostFunction __host__
    #define CudaGlobalFunction __global__
  #endif

  #ifndef CROSS_CPU
    #define CudaConstantMemory __constant__
    #define CudaExternConstantMemory(x)
    #define CudaSharedMemory __shared__
    #define CudaSyncThreads __syncthreads

    #define CudaKernelRun(a__,b__,c__,d__) a__<<<b__,c__>>>d__; HANDLE_ERROR( cudaThreadSynchronize()); HANDLE_ERROR( cudaGetLastError() )
    #ifdef CROSS_SYNC
      #define CudaKernelRunNoWait(a__,b__,c__,d__,e__) a__<<<b__,c__>>>d__; HANDLE_ERROR( cudaThreadSynchronize()); HANDLE_ERROR( cudaGetLastError() );
    #else
      #define CudaKernelRunNoWait(a__,b__,c__,d__,e__) a__<<<b__,c__,0,e__>>>d__;
    #endif      
//    #define CudaKernelRun(a__,b__,c__,d__) {dim3 _b_(b__.x,b__.y,1); int max_z_ = b__.z; for (int _z_=0; _z_<max_z_; _z_++) a__<<<_b_,c__>>>d__;}
    #define CudaBlock blockIdx
    #define CudaThread threadIdx
    #define CudaNumberOfThreads blockDim
    #define CudaCopyToConstant(a__,b__,c__,d__) HANDLE_ERROR( cudaMemcpyToSymbol(a__, c__, d__, 0, cudaMemcpyHostToDevice))
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
    #define CudaMallocHost(a__,b__) HANDLE_ERROR( cudaMallocHost(a__,b__) )
    #define CudaFree(a__) HANDLE_ERROR( cudaFree(a__) )
    #define CudaFreeHost(a__) HANDLE_ERROR( cudaFreeHost(a__) )

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

    #define CudaThreadSynchronize() HANDLE_ERROR( cudaThreadSynchronize() )

    #define CudaSetDevice(a__) HANDLE_ERROR( cudaSetDevice( a__ ) )
    #define CudaGetDeviceCount(a__) HANDLE_ERROR( cudaGetDeviceCount( a__ ) )

void HandleError( cudaError_t err, const char *file, int line );
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))
int GetMaxThreads();
#define RunKernelMaxThreads (GetMaxThreads())

__device__ int lock = 0;
__device__ inline void atomicAddP(type_f* a, type_f b)
{
        while(atomicCAS(&lock, 0, 1)) {};
        a[0] += b;
        lock = 0;
}


__shared__ type_f sumtab[512];

__device__ inline void atomicSum(type_f * sum, type_f val)
{
        int i = blockDim.x*blockDim.y;
        int k = blockDim.x*blockDim.y;
        int j = blockDim.x*threadIdx.y + threadIdx.x;
        sumtab[j] = val;
        __syncthreads();
        while (i> 1) {
                k = i >> 1;
                i = i - k;
                if (j<k) sumtab[j] += sumtab[j+i];
                __syncthreads();
        }
        if (j==0) atomicAddP(sum,sumtab[0]);
}
                                                                                                                                
  #else
    #include <assert.h>
    #include <time.h>
    #include <malloc.h>
    #include <cstdlib>
    #include <cstring>
    #include <math.h>
            
    #define CudaDeviceFunction
    #define CudaHostFunction
    #define CudaGlobalFunction
    #define CudaConstantMemory
    #define CudaExternConstantMemory(x) extern x
    #define CudaSharedMemory static
    #define CudaSyncThreads() //assert(CpuThread.x == 0)
    #define CudaKernelRun(a__,b__,c__,d__) for (CpuBlock.x = 0; CpuBlock.x < b__.x; CpuBlock.x++) \
                                        for (CpuBlock.y = 0; CpuBlock.y < b__.y; CpuBlock.y++) \
                                         for (CpuThread.x = 0; CpuThread.x < c__.x; CpuThread.x++) \
                                          for (CpuThread.y = 0; CpuThread.y < c__.y; CpuThread.y++) a__ d__
    #define CudaKernelRunNoWait(a__,b__,c__,d__,e__) CudaKernelRun(a__,b__,c__,d__);
    #define CudaBlock CpuBlock
    #define CudaThread CpuThread
    #define CudaNumberOfThreads CpuSize
    #define CudaCopyToConstant(a__,b__,c__,d__) std::memcpy(&b__, c__, d__)
    #define CudaMemcpy2D(a__,b__,c__,d__,e__,f__,g__) memcpy2D(a__, b__, c__, d__, e__, f__)
    #define CudaMemcpy(a__,b__,c__,d__) memcpy(a__, b__, c__)
    #define CudaMemcpyAsync(a__,b__,c__,d__,e__) CudaMemcpy(a__, b__, c__, d__)
    #define CudaMemset(a__,b__,c__) memset(a__, b__, c__)
    #define CudaMalloc(a__,b__) assert( (*((void**)(a__)) = malloc(b__)) != NULL )
    #define CudaMallocHost(a__,b__) assert( (*((void**)(a__)) = malloc(b__)) != NULL )
    #define CudaFree(a__) free(a__)
    #define CudaFreeHost(a__) free(a__)


    #define CudaEvent_t clock_t
    #define CudaEventCreate(a__) *a__ = 0
    #define CudaEventDestroy(a__)
    #define CudaEventRecord(a__,b__) a__ = b__
    #define CudaEventSynchronize(a__) a__ = clock()
    #define CudaEventElapsedTime(a__,b__,c__) *(a__) = (1000*((float)(c__ -  b__)))/CLOCKS_PER_SEC
    #define CudaThreadSynchronize()

    #define CudaStream_t int
    #define CudaStreamCreate(a__)
    #define CudaStreamSynchronize(a__)

    #define CudaThreadSynchronize()

    #define CudaSetDevice(a__) CpuSize.x=1;CpuSize.y=1;CpuSize.z=1;
    #define CudaGetDeviceCount(a__) *a__ = 1;

    #define RunKernelMaxThreads 1
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
    };
    struct uchar4 { unsigned char x,y,z,w; };

    extern uint3 CpuBlock;
    extern uint3 CpuThread;
    extern uint3 CpuSize;
    
    void memcpy2D(void * dst_, int dpitch, void * src_, int spitch, int width, int height);

    inline void atomicSum(float * sum, float val)
    {
      sum[0] += val;
    }


  #endif
#endif
#define CROSS_H
    
