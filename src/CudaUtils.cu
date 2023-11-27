#include "Consts.h"
#include "CudaUtils.hpp"
#include "StorageConversions.h"

namespace detail {
template <typename T>
struct FillExecutor : LinearExecutor {
    T* dev_ptr;
    T value;

    CudaDeviceFunction void Execute() const {
        const size_t i = CudaBlock.x * CudaNumberOfThreads.x + CudaThread.x;
        if (i < size) dev_ptr[i] = value;
    }
};

struct StorageNaNFillExecutor : LinearExecutor {
    storage_t* dev_ptr;

    CudaDeviceFunction void Execute() const {
        const size_t i = CudaBlock.x * CudaNumberOfThreads.x + CudaThread.x;
        const storage_t nan = real_to_storage(getStorageNaN());
        if (i < size) dev_ptr[i] = nan;
    }
};
}  // namespace detail

template <typename T>
void CudaFillN(T* device_ptr, size_t N, T value) {
    detail::FillExecutor<T> fill_exec{N, device_ptr, value};
    LaunchExecutor(fill_exec);
}

template <typename T>
void CudaFillNAsync(T* device_ptr, size_t N, T value, CudaStream_t stream) {
    detail::FillExecutor<T> fill_exec{N, device_ptr, value};
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

void fillWithStorageNaN(storage_t* device_ptr, size_t N) {
    detail::StorageNaNFillExecutor exec{N, device_ptr};
    LaunchExecutor(exec);
}

void fillWithStorageNaNAsync(storage_t* device_ptr, size_t N, CudaStream_t stream) {
    detail::StorageNaNFillExecutor exec{N, device_ptr};
    LaunchExecutorAsync(exec, stream);
}
