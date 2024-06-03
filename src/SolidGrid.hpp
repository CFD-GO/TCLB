#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <time.h>
#include <vector>
#include <set>
#include <math.h>
#include "SolidGrid.h"

template <class BALLS>
int SolidGrid<BALLS>::TryDepth(size_t grid_size) {
    // printf("Trying depth: %d\n", depth);
    size_t data_size = grid_size * depth;
    data.resize(data_size);
    for (size_t i=0; i<data.size(); i++) data[i] = -1;
    for (size_t i=0; i<balls->size(); i++) {
        size_t data_offset = 0;
        for (int k=0; k<3; k++) {
            double val = balls->getPos(i,k);
            int p = floor(val/delta);
            data_offset = data_offset * (maxs[k]-mins[k]+1) + p - mins[k];
        }
        data_offset = data_offset * depth;
        int k = 0;
        while (data[data_offset + k] != -1) {
            k++;
            if (k >= depth) {
                return -1;
            }
        }
        data[data_offset + k] = i;
    }
    return 0;
}

template <class BALLS>
void SolidGrid<BALLS>::Build () {
    if (balls->size() > 0) {
        double maxr = 0.5;
        if (depth < 1) depth = 4;
        for (size_t i=0; i<balls->size(); i++) {
            double val = balls->getRad(i);
            if (maxr < val) maxr = val;
        }
        delta = 2*maxr;
        // printf("delta: %lf\n", delta);
        for (int k=0; k<3; k++) {
            mins[k] = 0xFFFFFF;
            maxs[k] = -0xFFFFFF;
        }
        for (size_t i=0; i<balls->size(); i++) {
            for (int k=0; k<3; k++) {
                double val = balls->getPos(i,k);
                int p = floor(val/delta);
                if (mins[k] > p) mins[k] = p;
                if (maxs[k] < p) maxs[k] = p;
            }
        }
        size_t grid_size = 1;
        for (int k=0; k<3; k++) grid_size = grid_size * (maxs[k]-mins[k]+1);
        while (TryDepth(grid_size)) {
            depth = depth*2;
            output("Too many particles per bin in SolidGrid. Increasing bin size to %d\n", depth);
            if (depth > 1024) {
                ERROR("Too large SolidGrid bin size (hardcoded limit)\n");
                exit(-1);
            }
        }
    } else {
        delta = 1.0;
        depth = 0;
        data.resize(0);
        for (int k=0; k<3; k++) {
            mins[k] = 1;
            maxs[k] = 0;
        }
    }
}


template <class BALLS>
void SolidGrid<BALLS>::InitFinder (typename SolidGrid<BALLS>::finder_t& finder) {
    data_size_max = 0;
	finder.data = NULL;
    for (int k=0;k<3;k++) {
        finder.maxs[k] = -1;
        finder.mins[k] = 0;
    }
    finder.depth = 0;
    finder.delta = 1.0;
}

template <class BALLS>
void SolidGrid<BALLS>::CleanFinder (typename SolidGrid<BALLS>::finder_t& finder) {
	if (finder.data != NULL) CudaFree(finder.data);
    InitFinder(finder);
}


template <class BALLS>
void SolidGrid<BALLS>::CopyToGPU (typename SolidGrid<BALLS>::finder_t& finder, CudaStream_t stream) {
    if (data.size() > data_size_max) {
        if (finder.data != NULL) CudaFree(finder.data);
        data_size_max = data.size();
        CudaMalloc(&finder.data, data.size() * sizeof(gr_addr_t));
    }
    for (int k=0;k<3;k++) {
        finder.maxs[k] = maxs[k];
        finder.mins[k] = mins[k];
    }
    finder.depth = depth;
    finder.delta = delta;
    if (data.size() > 0) {
        CudaMemcpyAsync(finder.data, (gr_addr_t*) &data[0], data.size() * sizeof(gr_addr_t), CudaMemcpyHostToDevice, stream);
    }
}
