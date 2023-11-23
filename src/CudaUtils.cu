#include "Consts.h"
#include "CudaUtils.hpp"

namespace detail {
template <typename T>
struct FillExecutor {
    T* dev_ptr;
    size_t size;
    T value;

    CudaDeviceFunction void Execute() const {
        const size_t i = CudaBlock.x * CudaNumberOfThreads.x + CudaThread.x;
        if (i < size) dev_ptr[i] = value;
    }
    LaunchParams ComputeLaunchParams(dim3 max_threads) const {
        const size_t max_threads_per_block = max_threads.x * max_threads.y * max_threads.z;
        const size_t blocks_needed = (size + max_threads_per_block - 1) / max_threads_per_block;
        dim3 blocks, threads;
        blocks.x = blocks_needed;
        threads.x = max_threads_per_block;
        return {blocks, threads};
    }
};
}  // namespace detail

template <typename T>
void CudaFillN(T* device_ptr, size_t N, T value) {
    detail::FillExecutor<T> fill_exec{device_ptr, N, value};
    LaunchExecutor(fill_exec);
}

template <typename T>
void CudaFillNAsync(T* device_ptr, size_t N, T value, CudaStream_t stream) {
    detail::FillExecutor<T> fill_exec{device_ptr, N, value};
    LaunchExecutorAsync(fill_exec, stream);
}

#define INSTANTIATE_FOR_TYPE(type__)                                              \
    template void CudaFillN<type__>(type__ * device_ptr, size_t N, type__ value); \
    template void CudaFillNAsync<type__>(type__ * device_ptr, size_t N, type__ value, CudaStream_t stream);

INSTANTIATE_FOR_TYPE(int)
INSTANTIATE_FOR_TYPE(unsigned)
INSTANTIATE_FOR_TYPE(long)
INSTANTIATE_FOR_TYPE(unsigned long)
INSTANTIATE_FOR_TYPE(float)
INSTANTIATE_FOR_TYPE(double)
