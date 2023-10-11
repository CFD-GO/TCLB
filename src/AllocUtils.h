/** \file AllocUtils.h
*/

#ifndef ALLOCUTILS_H
#define ALLOCUTILS_H

#define ALLOCPRINT1 debug2("Allocating: %ld b\n", size)
#define ALLOCPRINT2 debug1("got address: (%p - %p)\n", tmp, (unsigned char*)tmp+size)

/// Allocation of a GPU memory Buffer
inline void * BAlloc(size_t size) {
  char * tmp = NULL;
  ALLOCPRINT1;
#ifdef DIRECT_MEM
  CudaMallocHost( (void**)&tmp, size );
#else
  CudaMalloc( (void**)&tmp, size );
#endif
  ALLOCPRINT2;
  CudaMemset( tmp, 0, size );
  return (void *) tmp;
}

/// Preallocation of a buffer (combines allocation into one big allocation)
inline void BPreAlloc(void ** ptr, size_t size) {
  CudaMalloc( ptr, size );
}

#endif // ALLOCUTILS_H
