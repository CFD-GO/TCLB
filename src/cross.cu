#include <stdio.h>
#include "cross.h"
#include "Global.h"
#include "Node.h"
#include "LatticeContainer.h"
#ifdef CROSS_CPU

uint3 CpuBlock, CpuThread, CpuSize;

void memcpy2D(void * dst_, int dpitch, void * src_, int spitch, int width, int height) {
	char * dst = (char*) dst_, *src = (char*) src_;
	for (int i=0; i<height; i++) {
		memcpy(dst + i*dpitch, src + i*spitch, width);
	}
}

#else

// Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        fprintf(stderr, "[%d] %s in %s at line %d\n", D_MPI_RANK, cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}

int GetMaxThreads()
{
            cudaFuncAttributes * attr = new cudaFuncAttributes;
            HANDLE_ERROR( cudaFuncGetAttributes(attr, RunKernel<Node>) );
            printf( "Constant mem:%ld\n", attr->constSizeBytes);
            printf( "Local    mem:%ld\n", attr->localSizeBytes);
            printf( "Max  threads:%d\n", attr->maxThreadsPerBlock);
            printf( "Reg   Number:%d\n", attr->numRegs);
            printf( "Shared   mem:%ld\n", attr->sharedSizeBytes);
            return attr->maxThreadsPerBlock;
}

#endif