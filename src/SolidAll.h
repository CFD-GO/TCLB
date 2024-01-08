#ifndef SOLIDALL_H

#include "types.h"

typedef int addr_t;

template <class BALLS>
class SolidAll {
public:
    class finder_t;
    template <class T>
    class set_found_t {
        const finder_t& finder;
        real_t* particle_data;
        real_t point[3];
        class iterator_t {
            const set_found_t * set;
            addr_t i;
            CudaDeviceFunction iterator_t(const set_found_t& set_, const addr_t& i_) : set(&set_), i(i_) { };
            friend class set_found_t;
        public:
            CudaDeviceFunction T operator* () { return T(set->particle_data, i, set->point); }
            CudaDeviceFunction iterator_t& operator++() { 
                ++i;
	            return *this;
            }
            CudaDeviceFunction bool operator!=(const iterator_t& other) { return i != other.i;};
        };
        public:
            typedef iterator_t iterator;
            CudaDeviceFunction inline iterator begin() { return iterator(*this, 0); };
            CudaDeviceFunction inline iterator end()   { return iterator(*this, finder.size); };
            CudaDeviceFunction inline set_found_t(const finder_t& finder_, real_t* particle_data_, const real_t point_[], const real_t lower[], const real_t upper[]) : finder(finder_),  particle_data(particle_data_) {
                for (int i=0;i<3;i++) { point[i]=point_[i]; }
            };
    };
    class finder_t {
        size_t size;
        friend class SolidAll<BALLS>;
    public:
        template <class T>
        CudaDeviceFunction inline set_found_t<T> find(const real_t point[], const real_t lower[], const real_t upper[]) const {
            return set_found_t<T>(*this, point, lower, upper);
        };
    };
private:
    size_t part_size;
public:
    BALLS* balls;
    void Build() {
        part_size = balls->size();
    }
    void InitFinder(finder_t& f) { f.size = 0; };
    void CleanFinder(finder_t& f) { };
    void CopyToGPU(finder_t& f, CudaStream_t stream) { f.size = part_size; };
};

#define SOLIDALL_H
#endif
