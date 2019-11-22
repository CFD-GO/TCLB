#ifndef PINNED_ALLOCATOR_H
#define PINNED_ALLOCATOR_H

#include <memory>
#include <stdio.h>

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
                        fprintf(stderr, "Alloc %ld bytes.\n", n*sizeof(T));
                        return std::allocator<T>::allocate(n, hint);
                }

                void deallocate(pointer p, size_type n)
                {
                        fprintf(stderr, "Dealloc %ld bytes (%p).\n", n*sizeof(T), p);
                        return std::allocator<T>::deallocate(p, n);
                }

                pinned_allocator() throw(): std::allocator<T>() { fprintf(stderr, "Hello allocator!\n"); }
                pinned_allocator(const pinned_allocator &a) throw(): std::allocator<T>(a) { }
                template <class U>                    
                pinned_allocator(const pinned_allocator<U> &a) throw(): std::allocator<T>(a) { }
                ~pinned_allocator() throw() { }
        };

#endif
