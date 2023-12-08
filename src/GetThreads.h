#ifndef THREADS_H
#define THREADS_H

#include <typeinfo>

#include "Global.h"
#include "cross.h"

std::string cxx_demangle(std::string str);

inline int ceiling_div(int x, int y) {
    return x / y + (x % y != 0);
}

template <class EX>
CudaGlobalFunction void Kernel(const EX executor) {
    executor.Execute();
}

/// Get maximum number of threads for all kernels at runtime
template <class EX>
int GetThreads() {
    CudaFuncAttributes attr;
    auto attr_ptr = &attr;
    CudaFuncGetAttributes(attr_ptr, Kernel<EX>);
    debug1("[%d] Constant mem:%ld\n", D_MPI_RANK, attr.constSizeBytes);
    debug1("[%d] Local    mem:%ld\n", D_MPI_RANK, attr.localSizeBytes);
    debug1("[%d] Max  threads:%d\n", D_MPI_RANK, attr.maxThreadsPerBlock);
    debug1("[%d] Reg   Number:%d\n", D_MPI_RANK, attr.numRegs);
    debug1("[%d] Shared   mem:%ld\n", D_MPI_RANK, attr.sharedSizeBytes);
    return attr.maxThreadsPerBlock;
}

class ThreadNumberCalculatorBase {
    typedef ThreadNumberCalculatorBase type;
    typedef std::vector<type*> list_t;
    static inline list_t& List() {
        static list_t list;
        return list;
    }
    static inline bool compare(const type* a, const type* b) { return a->name < b->name; }

   protected:
    dim3 thr;
    unsigned int maxthr;
    std::string name;

   public:
    static void InitAll();
    ThreadNumberCalculatorBase();
    virtual void Init() = 0;
    inline dim3 threads() { return thr; }
    void print();
};

template <class EX>
class ThreadNumberCalculator : public ThreadNumberCalculatorBase {
   public:
    virtual void Init() {
        name = cxx_demangle(typeid(EX).name());
        maxthr = GetThreads<EX>();
        thr.z = 1;
        int val = maxthr;
        if (maxthr < X_BLOCK) {
            thr.x = maxthr;
            thr.y = 1;
        } else {
            if (val > MAX_THREADS) { val = MAX_THREADS; }
            thr.x = X_BLOCK;
            thr.y = val / X_BLOCK;
        }
    };
};

template <class EX>
class ThreadNumber {
    typedef ThreadNumberCalculator<EX> calc_t;
    static calc_t calc;

   public:
    static inline dim3 threads() { return calc.threads(); }
};

template <class EX>
ThreadNumberCalculator<EX> ThreadNumber<EX>::calc;

/// Initialize Thread/Block number variables
int InitDim();

struct LaunchParams {
    dim3 blx, thr;
};

template <class EX>
LaunchParams ComputeLaunchParams(const EX& executor) {
    const auto threads = ThreadNumber<EX>::threads();
    return executor.ComputeLaunchParams(threads);
}

template <class EX>
void LaunchExecutor(const EX& executor) {
    const auto exec_params = ComputeLaunchParams(executor);
    debug1("Launching kernel: blocks: %dx%dx%d; threads: %dx%dx%d;", exec_params.blx.x, exec_params.blx.y, exec_params.blx.z, exec_params.thr.x, exec_params.thr.y, exec_params.thr.z);
    CudaKernelRun(Kernel<EX>, exec_params.blx, exec_params.thr, executor);
}

template <class EX>
void LaunchExecutorAsync(const EX& executor, CudaStream_t stream) {
    const auto exec_params = ComputeLaunchParams(executor);
    debug1("Launching async kernel: blocks: %dx%dx%d; threads: %dx%dx%d; stream: %p", exec_params.blx.x, exec_params.blx.y, exec_params.blx.z, exec_params.thr.x, exec_params.thr.y, exec_params.thr.z, stream);
    CudaKernelRunAsync(Kernel<EX>, exec_params.blx, exec_params.thr, stream, executor);
}

/// Base class for executors operating on a linear iteration space
/// Computes LaunchParams so that there is enough blocks and threads to cover an iteration space of size `size`
/// Uses the maximum dim3 number of threads per block, and assigns blocks linearly along x
struct LinearExecutor {
    unsigned size;
    LaunchParams ComputeLaunchParams(dim3 max_threads) const {
        const unsigned max_threads_per_block = max_threads.x * max_threads.y * max_threads.z;
        const unsigned blocks_needed = std::max(1u, (size + max_threads_per_block - 1) / max_threads_per_block);
        dim3 blocks;
        blocks.x = blocks_needed;
        return {blocks, max_threads};
    }

   protected:
    /// Get the linear grid index of the current thread (You must pass in the grid params, since they're only available in device code)
    template <typename D1, typename D2, typename D3>  /// TODO: the template here is due to the fact that we use uint3 instead of dim3 on CPU. Investigate why and unify everything to dim3
    CudaDeviceFunction unsigned threadID(D1 thread, D2 block, D3 block_size) const {
        const auto threads_per_block = block_size.x * block_size.y * block_size.z;
        return thread.x + block_size.x * (thread.y + thread.z * block_size.y) + block.x * threads_per_block;
    }
    /// Check whether the current thread is within the execution range
    CudaDeviceFunction bool inRange(unsigned thread_id) const { return thread_id < size; }
};

#endif  // THREADS_H
