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
    CudaKernelRun(Kernel<EX>, exec_params.blx, exec_params.thr, executor);
}

template <class EX>
void LaunchExecutorAsync(const EX& executor, CudaStream_t stream) {
    const auto exec_params = ComputeLaunchParams(executor);
    CudaKernelRunAsync(Kernel<EX>, exec_params.blx, exec_params.thr, stream, executor);
}

#endif  // THREADS_H
