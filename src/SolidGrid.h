#ifndef SOLIDGRID_H

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <time.h>
#include <vector>
#include <set>
#include <math.h>

#include "types.h"
#include "Global.h"

typedef int gr_addr_t;

template <class BALLS>
class SolidGrid {
public:
    class finder_t;
    template <class T>
    class set_found_t {
        const finder_t& finder;
        real_t* particle_data;
        real_t point[3];
        int mins[3];
        int maxs[3];
        class iterator_t {
            const set_found_t * set;
            int idx[3];
            int balli;
            int d;
            CudaDeviceFunction size_t calc_data_offset() const {
                size_t data_offset = 0;
                for (int k=0; k<3; k++) data_offset = data_offset * (set->finder.maxs[k]-set->finder.mins[k]+1) + idx[k]-set->finder.mins[k];
                return data_offset * set->finder.depth;
            };
            CudaDeviceFunction void go() {
                while (idx[0] <= set->maxs[0]) {
                    while (idx[1] <= set->maxs[1]) {
                        while (idx[2] <= set->maxs[2]) {
                            size_t data_offset = calc_data_offset();
                            while (d < set->finder.depth) {
                                balli = set->finder.data[data_offset+d];
                                if (balli != -1) return;
                                d++;
                            }
                            d = 0;
                            idx[2]++;
                        }
                        idx[2]=set->mins[2];
                        idx[1]++;
                    }
                    idx[1]=set->mins[1];
                    idx[0]++;
                }
                balli = -1;
                return;
            };
            CudaDeviceFunction iterator_t(const set_found_t& set_) : set(&set_) {
                for (int i=0; i<3; i++) idx[i] = set->mins[i];
                d = 0;
                go();
            };
            CudaDeviceFunction iterator_t() : set(NULL) { balli = -1; };
            friend class set_found_t;
        public:
            CudaDeviceFunction T operator* () { return T(set->particle_data, balli, set->point); }
            CudaDeviceFunction iterator_t& operator++() { 
                ++d;
                go();
	            return *this;
            }
            CudaDeviceFunction bool operator!=(const iterator_t& other) { return balli != other.balli; };
        };
        public:
            typedef iterator_t iterator;
            CudaDeviceFunction inline iterator begin() { return iterator(*this); };
            CudaDeviceFunction inline iterator end()   { return iterator(); };
            CudaDeviceFunction inline set_found_t(const finder_t& finder_, real_t* particle_data_, const real_t point_[], const real_t lower[], const real_t upper[]): finder(finder_),  particle_data(particle_data_) {
                for (int i=0;i<3;i++) { point[i]=point_[i]; }
                real_t d = 0.5*finder.delta;
                for (int k=0; k<3; k++) {
                    mins[k] = floor((lower[k]-d)/finder.delta);
                    if (mins[k] < finder.mins[k]) mins[k] = finder.mins[k];
                    maxs[k] = floor((upper[k]+d)/finder.delta);
                    if (maxs[k] > finder.maxs[k]) maxs[k] = finder.maxs[k];
                }
            };
    };
    template <class T, int MAX_CACHE>
    class cache_set_found_t {
        real_t* particle_data;
        real_t point[3];
        size_t cache_size;
        tr_addr_t cache[MAX_CACHE];
        class iterator_t {
            const cache_set_found_t * set;
            size_t i;
            CudaDeviceFunction iterator_t(const cache_set_found_t& set_, const size_t& i_) : set(&set_), i(i_) { };
            friend class cache_set_found_t;
        public:
            CudaDeviceFunction T operator* () { return T(set->particle_data, set->cache[i], set->point); }
            CudaDeviceFunction iterator_t& operator++() { 
                ++i;
	            return *this;
            }
            CudaDeviceFunction bool operator!=(const iterator_t& other) { return i != other.i; };
        };
        public:
            typedef iterator_t iterator;
            CudaDeviceFunction inline iterator begin() { return iterator(*this, 0); };
            CudaDeviceFunction inline iterator end()   { return iterator(*this, cache_size); };
            CudaDeviceFunction inline cache_set_found_t(const finder_t& finder, real_t* particle_data_, const real_t point_[], const real_t lower[], const real_t upper[]) : particle_data(particle_data_) {
                for (int i=0;i<3;i++) { point[i]=point_[i]; }
                int mins[3];
                int maxs[3];
                real_t d = 0.5*finder.delta;
                for (int k=0; k<3; k++) {
                    mins[k] = floor((lower[k]-d)/finder.delta);
                    if (mins[k] < finder.mins[k]) mins[k] = finder.mins[k];
                    maxs[k] = floor((upper[k]+d)/finder.delta);
                    if (maxs[k] > finder.maxs[k]) maxs[k] = finder.maxs[k];
                }
                //  for (int k=0; k<3; k++) {
                //      printf("pos[%d]/%f: %f -> range[%d]: %d - %d\n", k, finder.delta, point[k]/finder.delta, k, mins[k], maxs[k]);
                //  }
                int idx[3];
                cache_size = 0;
                for (idx[0]=mins[0]; idx[0]<=maxs[0]; idx[0]++)
                for (idx[1]=mins[1]; idx[1]<=maxs[1]; idx[1]++)
                for (idx[2]=mins[2]; idx[2]<=maxs[2]; idx[2]++) {
                    size_t data_offset = 0;
                    for (int k=0; k<3; k++) data_offset = data_offset * (finder.maxs[k]-finder.mins[k]+1) + idx[k]-finder.mins[k];
                    data_offset = data_offset * finder.depth;
                    for (int k=0;k<finder.depth;k++) {
                        // printf("%d %d %d %d -> %d\n",idx[0],idx[1],idx[2],k, finder.data[data_offset + k]);
                        if (finder.data[data_offset + k] == -1) break;
                        cache[cache_size] = finder.data[data_offset + k];
                        ++cache_size;
                        if (cache_size >= MAX_CACHE) { return; }
                    }
                }
            };
    };
    class finder_t {
        int mins[3];
        int maxs[3];
        int depth;
        real_t delta;
        gr_addr_t* data;
        friend class SolidGrid<BALLS>;
    public:
        template <class T>
        CudaDeviceFunction inline set_found_t<T> find(const real_t point[], const real_t lower[], const real_t upper[]) const {
            return set_found_t<T>(*this, point, lower, upper);
        };
        template <class T, int MAX_CACHE>
        CudaDeviceFunction inline cache_set_found_t<T, MAX_CACHE> cache_find(const real_t point[], const real_t lower[], const real_t upper[]) const {
            return cache_set_found_t<T, MAX_CACHE>(*this, point, lower, upper);
        };
    };
private:
    int mins[3];
    int maxs[3];
    int depth = 0;
    real_t delta;
    std::vector<gr_addr_t> data;
    size_t data_size_max;
    inline int TryDepth(size_t grid_size);
public:
    BALLS* balls;
    inline void Build();
    inline void InitFinder(finder_t&);
    inline void CleanFinder(finder_t&);
    inline void CopyToGPU(finder_t&, CudaStream_t stream);
};

template <class BALLS>
int SolidGrid<BALLS>::TryDepth(size_t grid_size) {
    // printf("Trying depth: %d\n", depth);
    size_t data_size = grid_size * depth;
    data.resize(data_size);
    for (size_t i = 0; i < data.size(); i++) data[i] = -1;
    for (size_t i = 0; i < balls->size(); i++) {
        size_t data_offset = 0;
        for (int k = 0; k < 3; k++) {
            double val = balls->getPos(i, k);
            int p = floor(val / delta);
            data_offset = data_offset * (maxs[k] - mins[k] + 1) + p - mins[k];
        }
        data_offset = data_offset * depth;
        int k = 0;
        while (data[data_offset + k] != -1) {
            k++;
            if (k >= depth) { return -1; }
        }
        data[data_offset + k] = i;
    }
    return 0;
}

template <class BALLS>
void SolidGrid<BALLS>::Build() {
    if (balls->size() > 0) {
        double maxr = 0.5;
        if (depth < 1) depth = 4;
        for (size_t i = 0; i < balls->size(); i++) {
            double val = balls->getRad(i);
            if (maxr < val) maxr = val;
        }
        delta = 2 * maxr;
        // printf("delta: %lf\n", delta);
        for (int k = 0; k < 3; k++) {
            mins[k] = 0xFFFFFF;
            maxs[k] = -0xFFFFFF;
        }
        for (size_t i = 0; i < balls->size(); i++) {
            for (int k = 0; k < 3; k++) {
                double val = balls->getPos(i, k);
                int p = floor(val / delta);
                if (mins[k] > p) mins[k] = p;
                if (maxs[k] < p) maxs[k] = p;
            }
        }
        size_t grid_size = 1;
        for (int k = 0; k < 3; k++) grid_size = grid_size * (maxs[k] - mins[k] + 1);
        depth = 4;
        while (TryDepth(grid_size)) {
            depth = depth * 2;
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
        for (int k = 0; k < 3; k++) {
            mins[k] = 1;
            maxs[k] = 0;
        }
    }
}

template <class BALLS>
void SolidGrid<BALLS>::InitFinder(finder_t& finder) {
    data_size_max = 0;
    finder.data = NULL;
    for (int k = 0; k < 3; k++) {
        finder.maxs[k] = -1;
        finder.mins[k] = 0;
    }
    finder.depth = 0;
    finder.delta = 1.0;
}

template <class BALLS>
void SolidGrid<BALLS>::CleanFinder(finder_t& finder) {
    if (finder.data != NULL) CudaFree(finder.data);
    InitFinder(finder);
}

template <class BALLS>
void SolidGrid<BALLS>::CopyToGPU(finder_t& finder, CudaStream_t stream) {
    if (data.size() > data_size_max) {
        if (finder.data != NULL) CudaFree(finder.data);
        data_size_max = data.size();
        CudaMalloc(&finder.data, data.size() * sizeof(gr_addr_t));
    }
    for (int k = 0; k < 3; k++) {
        finder.maxs[k] = maxs[k];
        finder.mins[k] = mins[k];
    }
    finder.depth = depth;
    finder.delta = delta;
    if (data.size() > 0) { CudaMemcpyAsync(finder.data, (gr_addr_t*)&data[0], data.size() * sizeof(gr_addr_t), CudaMemcpyHostToDevice, stream); }
}

#define SOLIDGRID_H
#endif
