#ifndef PINNED_ALLOCATOR_H
#define PINNED_ALLOCATOR_H

#include <memory>
#include <stdio.h>
#include "cross.h"

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

#endif
