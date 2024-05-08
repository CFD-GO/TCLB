#ifndef SOLIDGRID_H

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <time.h>
#include <vector>
#include <set>
#include <math.h>

typedef int gr_addr_t;



template <class BALLS>
class SolidGrid {
public:
    class finder_t;
    template <class T>
    class set_found_t {
        const finder_t& finder;
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
            CudaDeviceFunction T operator* () { return T(balli,set->point); }
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
            CudaDeviceFunction inline set_found_t(const finder_t& finder_, const real_t point_[], const real_t lower[], const real_t upper[]): finder(finder_) {
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
        real_t point[3];
        size_t cache_size;
        tr_addr_t cache[MAX_CACHE];
        class iterator_t {
            const cache_set_found_t * set;
            size_t i;
            CudaDeviceFunction iterator_t(const cache_set_found_t& set_, const size_t& i_) : set(&set_), i(i_) { };
            friend class cache_set_found_t;
        public:
            CudaDeviceFunction T operator* () { return T(set->cache[i],set->point); }
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
            CudaDeviceFunction inline cache_set_found_t(const finder_t& finder, const real_t point_[], const real_t lower[], const real_t upper[]) {
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
    int depth;
    real_t delta;
    std::vector<gr_addr_t> data;
    size_t data_size_max;
    int TryDepth(size_t grid_size);
public:
    BALLS* balls;
    inline SolidGrid() {
        depth = 4;
    }
    void Build();
    void InitFinder(finder_t&);
    void CleanFinder(finder_t&);
    void CopyToGPU(finder_t&, CudaStream_t stream);
};

#define SOLIDGRID_H
#endif
