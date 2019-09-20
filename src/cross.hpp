#ifndef CROSS_CPU

  __shared__ real_t  sumtab[MAX_THREADS];
  __shared__ real_t* sumptr[MAX_THREADS];

  #if __CUDA_ARCH__ < 200
    #define CROSS_NEED_ATOMICADD
  #endif
      
  #ifdef CALC_DOUBLE_PRECISION
    typedef unsigned long long int real_t_i;
    #define R2I(x) __double_as_longlong(x)
    #define I2R(x) __longlong_as_double(x)
    #define CROSS_NEED_ATOMICADD
  #else
    #define R2I(x) __float_as_int(x)
    #define I2R(x) __int_as_float(x)
    typedef int      real_t_i;
  #endif
        
  #ifdef CROSS_NEED_ATOMICADD
      __device__ inline void atomicAddP(real_t* address, real_t val)
      {
          if (val != 0.0) {
            real_t_i* address_as_ull = (real_t_i*) address;
            real_t_i  old = *address_as_ull;
            real_t_i  assumed, nw;
            do {
                assumed = old;
                nw = R2I(val + I2R(assumed));
                old = atomicCAS(address_as_ull, assumed, nw);
            } while (assumed != old);
          }
      }
   #else
      #define atomicAddP atomicAdd
   #endif

        
    __device__ inline void atomicMaxP(real_t* address, real_t val)
    {
        if (val != 0.0) {
          real_t_i* address_as_ull = (real_t_i*) address;
          real_t_i  old = *address_as_ull;
          real_t_i  assumed, nw;
          do {
              assumed = old;
              nw = R2I(max(val,I2R(assumed)));
              old = atomicCAS(address_as_ull, assumed, nw);
          } while (assumed != old);
        }
    }
    #ifndef MAX_THREADS
      #error FUCK!
    #else
      #if MAX_THREADS < 500
        #error Double Fuck!
      #endif
    #endif

      __device__ inline real_t blockSum(real_t val) {
              int i = blockDim.x*blockDim.y;
              int k = blockDim.x*blockDim.y;
              int j = blockDim.x*threadIdx.y + threadIdx.x;
              sumtab[j] = val;
              while (i> 1) {
                      k = i >> 1;
                      i = i - k;
                      if (j<k) sumtab[j] += sumtab[j+i];
                      __syncthreads();
              }
              return sumtab[0];
      }
      
      __device__ inline void atomicSum_f(real_t * sum) {
              int i = blockDim.x*blockDim.y;
              int k = blockDim.x*blockDim.y;
              int j = blockDim.x*threadIdx.y + threadIdx.x;
              while (i> 1) {
                      k = i >> 1;
                      i = i - k;
                      if (j<k) sumtab[j] += sumtab[j+i];
                      __syncthreads();
              }
              if (j==0) {
                real_t val = sumtab[0];
                if (val != 0.0) {
                  atomicAddP(sum, val);
                }
              }
      }

      __device__ inline void atomicSum(real_t * sum, real_t val)
      {
              __syncthreads();
              int j = blockDim.x*threadIdx.y + threadIdx.x;
              sumtab[j] = val;
              __syncthreads();
              atomicSum_f(sum);
      }

/*      __device__ inline void atomicSum(real_t * sum, real_t val) {
        typedef cub::BlockReduce<real_t, 32, cub::BLOCK_REDUCE_WARP_REDUCTIONS, 20> BlockReduce;
        __shared__ typename BlockReduce::TempStorage temp_storage;
        real_t ret = BlockReduce(temp_storage).Sum(val);
        if ((blockDim.x*threadIdx.y + threadIdx.x) == 0) atomicAddP(sum, ret);
      }
*/

      __device__ inline void atomicMax(real_t * sum, real_t val)
      {
              int i = blockDim.x*blockDim.y;
              int k = blockDim.x*blockDim.y;
              int j = blockDim.x*threadIdx.y + threadIdx.x;
              __syncthreads();
              sumtab[j] = val;
              __syncthreads();
              while (i> 1) {
                      k = i >> 1;
                      i = i - k;
                      if (j<k) sumtab[j] = max(sumtab[j],sumtab[j+i]);
                      __syncthreads();
              }
              if (j==0) atomicMaxP(sum,sumtab[0]);
      }

      __device__ inline void atomicSumDiff(real_t * sum, real_t val, bool yes)
      {
                __syncthreads();
                int j = blockDim.x*threadIdx.y + threadIdx.x;
                if (yes) {
                  sumtab[j] = val;
                } else {
                  sumtab[j] = 0.0;
                }
                __syncthreads();
                atomicSum_f(sum);
      }

#endif
