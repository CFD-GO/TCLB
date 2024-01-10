#ifndef CUDAUTILS_HPP
#define CUDAUTILS_HPP

#include <cassert>
#include <memory>

#include "GetThreads.h"
#include "types.h"

namespace detail {
template <typename T>
struct CudaDeleter {
    void operator()(T* ptr) const { CudaFree(ptr); }
};
}  // namespace detail

// Memory owner of device memory
template <typename T>
using CudaUniquePtr = std::unique_ptr<T, detail::CudaDeleter<T>>;

// Semantics: allocate uninitialized array on device (as in C++20's std::make_unique_for_overwrite<T[]>)
template <typename T>
CudaUniquePtr<T> cudaMakeUnique(size_t size) {
    static_assert(std::is_trivial_v<T>, "Objects allocated on device must be of trivial type");

    void* ptr = nullptr;
    CudaMalloc(&ptr, size * sizeof(T));  // Alignment?
    if (!ptr) throw std::bad_alloc{};
    return CudaUniquePtr<T>(static_cast<T*>(ptr));
}

/// Allocate 2D row-major array with row padding to promote coalesced memory access. At least num_rows * num_cols * sizeof(T) will be allocated
/// \tparam T allocation type, must be trivial
/// \param num_cols number of columns required, this will be updated to reflect the padding (num_cols_pre <= num_cols_post)
/// \param num_rows number of rows
/// \return
template <typename T>
CudaUniquePtr<T> cudaMakeUnique2D(size_t& num_cols, size_t num_rows) {
    static_assert(std::is_trivial_v<T>, "Objects allocated on device must be of trivial type");

    void* ptr = nullptr;
    size_t row_sz_bytes = num_cols * sizeof(T);
    CudaMallocPitch(&ptr, &row_sz_bytes, row_sz_bytes, num_rows);
    assert(row_sz_bytes % sizeof(T) == 0);
    num_cols = row_sz_bytes / sizeof(T);
    if (!ptr) throw std::bad_alloc{};
    return CudaUniquePtr<T>(static_cast<T*>(ptr));
}

/// Copy std::vector to device
/// \tparam T data type
/// \tparam Alloc allocator type
/// \param device_ptr destination device pointer
/// \param vec source host vector
template <typename T, typename Alloc>
void copyVecToDevice(T* device_ptr, const std::vector<T, Alloc>& vec) {
    CudaMemcpy(device_ptr, vec.data(), vec.size() * sizeof(T), CudaMemcpyHostToDevice);
}

/// Copy std::vector to device asynchronously
/// \tparam T data type
/// \tparam Alloc allocator type
/// \param device_ptr destination device pointer
/// \param vec source host vector
/// \param stream execution stream
template <typename T, typename Alloc>
void copyVecToDeviceAsync(T* device_ptr, const std::vector<T, Alloc>& vec, CudaStream_t stream) {
    CudaMemcpyAsync(device_ptr, vec.data(), vec.size() * sizeof(T), CudaMemcpyHostToDevice, stream);
}

/// std::fill_n executed in device memory
/// \tparam T type of the array being filled
/// \param device_ptr pointer to beginning of memory region
/// \param N number of elements to write
/// \param value value to write
template <typename T>
void CudaFillN(T* device_ptr, unsigned N, T value);

/// std::fill_n executed asynchronously in device memory
/// \tparam T type of the array being filled
/// \param device_ptr pointer to beginning of memory region
/// \param N number of elements to write
/// \param value value to write
/// \param stream CUDA stream in which the fill will be executed
template <typename T>
void CudaFillNAsync(T* device_ptr, unsigned N, T value, CudaStream_t stream);

/// Fill array with storage NaNs. This needs special treatment, since we cannot obtain a NaN value on host for fp16.
void fillWithStorageNaN(storage_t* device_ptr, unsigned N);

/// Fill array with storage NaNs. This needs special treatment, since we cannot obtain a NaN value on host for fp16.
void fillWithStorageNaNAsync(storage_t* device_ptr, unsigned N, CudaStream_t stream);

#endif  // CUDAUTILS_HPP
