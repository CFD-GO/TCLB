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
    class cache_set_found_t {
        real_t point[3];
        size_t cache_size;
        tr_addr_t cache[max_cache_size];
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
                int mins[3];
                int maxs[3];
                real_t d = 0.5*finder.delta;
                for (int k=0; k<3; k++) {
                    mins[k] = floor((lower[k]-d)/finder.delta);
                    if (mins[k] < finder.mins[k]) mins[k] = finder.mins[k];
                    maxs[k] = floor((upper[k]+d)/finder.delta);
                    if (maxs[k] < finder.maxs[k]) maxs[k] = finder.maxs[k];
                }
                int idx[3];
                cache_size = 0;
                for (idx[0]=mins[0]; idx[0]<=maxs[0]; idx[0]++)
                for (idx[1]=mins[1]; idx[1]<=maxs[1]; idx[1]++)
                for (idx[2]=mins[2]; idx[2]<=maxs[2]; idx[2]++) {
                    size_t data_offset = 0;
                    for (int k=0; k<3; k++) data_offset = data_offset * (finder.maxs[k]-finder.mins[k]+1) + idx[k];
                    data_offset = data_offset * finder.depth;
                    for (int k=0;k<finder.depth;k++) {
                        if (finder.data[k] == -1) break;
                        cache[cache_size] = finder.data[k];
                        ++cache_size;
                        if (cache_size >= max_cache_size) return;
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
        CudaDeviceFunction inline auto find(const real_t point[], const real_t lower[], const real_t upper[]) const {
            //return set_found_t<T>(*this, point, lower, upper);
            return cache_set_found_t<T>(*this, point, lower, upper);
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
    void Build();
    void InitFinder(finder_t&);
    void CleanFinder(finder_t&);
    void CopyToGPU(finder_t&, CudaStream_t stream);
};

#define SOLIDGRID_H
#endif
