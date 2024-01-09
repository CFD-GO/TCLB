#ifndef STORAGECONVERSIONS_H
#define STORAGECONVERSIONS_H

// Conversions for storage_t. Since this potentially involves fp16, it must be hidden from host code
#include "types.h"

#ifndef STORAGE_BITS
#define storage_to_real(x__) x__
#define real_to_storage(x__) x__
inline CudaHostFunction CudaDeviceFunction storage_t getStorageNaN() {
    return std::numeric_limits<storage_t>::signaling_NaN();
}
#elif STORAGE_BITS == 16
#include <cuda_fp16.h>
inline CudaDeviceFunction real_t storage_to_real(storage_t v) {
    return __short_as_half(v);
}
inline CudaDeviceFunction storage_t real_to_storage(real_t v) {
    return __half_as_short(v);
}
inline CudaHostFunction CudaDeviceFunction storage_t getStorageNaN() {
    return __float2half(std::numeric_limits<float>::signaling_NaN());
}
#elif STORAGE_BITS == 32
inline CudaDeviceFunction real_t storage_to_real(storage_t v) {
    return __int_as_float(v);
}
inline CudaDeviceFunction storage_t real_to_storage(real_t v) {
    return __float_as_int(v);
}
inline CudaHostFunction CudaDeviceFunction storage_t getStorageNaN() {
    return std::numeric_limits<storage_t>::signaling_NaN();
}
#elif STORAGE_BITS == 64
inline CudaDeviceFunction real_t storage_to_real(storage_t v) {
    return __longlong_as_double(v);
}
inline CudaDeviceFunction storage_t real_to_storage(real_t v) {
    return __double_as_longlong(v);
}
inline CudaHostFunction CudaDeviceFunction storage_t getStorageNaN() {
    return std::numeric_limits<storage_t>::signaling_NaN();
}
#else
#error "If defined, 'STORAGE_BITS' must be one of {16, 32, 64}"
#endif

#endif  // STORAGECONVERSIONS_H
