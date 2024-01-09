#ifndef PINNED_ALLOCATOR_HPP
#define PINNED_ALLOCATOR_HPP

#include <memory>
#include <stdio.h>
#include "cross.h"
#include "Global.h"

        template <typename T>
        class pinned_allocator: public std::allocator<T>
        {
public:
                typedef size_t size_type;
                typedef T* pointer;
                typedef const T* const_pointer;

                template<typename _Tp1>
                struct rebind
                {
                        typedef pinned_allocator<_Tp1> other;
                };

                pointer allocate(size_type n, const void *hint=0)
                {
                        pointer ptr;
        		CudaMallocHost(&ptr, n*sizeof(T));
                        return ptr;
                }

                void deallocate(pointer ptr, size_type n)
                {
                        CudaFreeHost(ptr);
                        return;
                }

                pinned_allocator() throw(): std::allocator<T>() { debug0("Creating pinned allocator...\n"); }
                pinned_allocator(const pinned_allocator &a) throw(): std::allocator<T>(a) { }
                template <class U>                    
                pinned_allocator(const pinned_allocator<U> &a) throw(): std::allocator<T>(a) { }
                ~pinned_allocator() throw() { }
        };

#include <memory_resource>
#ifdef __cpp_lib_memory_resource
#include <unordered_map>
#include <memory>
class PinnedMemoryResource : public std::pmr::memory_resource {
    // Maps (aligned) pointers returned from the allocator those originally obtained from cudaMallocHost
    std::unordered_map<void*, void*> ptr_map;

public:
    void* do_allocate(size_t bytes, size_t alignment) override {
        void* cuda_ptr = nullptr;
        size_t space = alignment + bytes;
        CudaMallocHost(&cuda_ptr, space); // Error handling baked in
        void* dummy_ptr = cuda_ptr;
        void* const aligned = std::align(alignment, bytes, dummy_ptr, space);
        if(aligned != cuda_ptr)
            ptr_map.emplace(aligned, cuda_ptr);
        return aligned;
    }
    void do_deallocate(void* ptr, size_t /* bytes */, size_t /* alignment */) override {
        const auto it = ptr_map.find(ptr);
        void* dealloc_ptr = nullptr;
        if(it != ptr_map.end()) {
            dealloc_ptr = it->second;
            ptr_map.erase(it);
        } else
            dealloc_ptr = ptr;
        CudaFreeHost(dealloc_ptr);
    }
    bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override {
        if(auto as_pinned_ptr = dynamic_cast<const PinnedMemoryResource*>(&other); as_pinned_ptr)
            return ptr_map == as_pinned_ptr->ptr_map;
        return false;
    }
};
inline PinnedMemoryResource global_pinned_resource;

#endif // __cpp_lib_memory_resource
#endif // PINNED_ALLOCATOR_HPP
