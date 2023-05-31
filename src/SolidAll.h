#ifndef SOLIDALL_H

typedef int addr_t;

template <class BALLS>
class SolidAll {
public:
    class finder_t;
    template <class T>
    class set_found_t {
        const finder_t& finder;
        real_t point[3];
        class iterator_t {
            const set_found_t * set;
            addr_t i;
            CudaDeviceFunction iterator_t(const set_found_t& set_, const addr_t& i_) : set(&set_), i(i_) { };
            friend class set_found_t;
        public:
            CudaDeviceFunction T operator* () { return T(i,set->point); }
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
            CudaDeviceFunction inline set_found_t(const finder_t& finder_, const real_t point_[]): finder(finder_) {
                for (int i=0;i<3;i++) { point[i]=point_[i]; }
            };
    };
    class finder_t {
        size_t size;
        friend class SolidAll<BALLS>;
    public:
        template <class T>
        CudaDeviceFunction inline set_found_t<T> find(const real_t point[], const real_t lower[], const real_t upper[]) const {
            return set_found_t<T>(*this, point);
        };
    };
private:
    size_t part_size;
public:
    BALLS* balls;
    inline void Build() {
        part_size = balls->size();
    }
    inline void InitFinder(finder_t& f) { f.size = 0; };
    inline void CleanFinder(finder_t& f) { };
    inline void CopyToGPU(finder_t& f, CudaStream_t stream) { f.size = part_size; };
};

#define SOLIDALL_H
#endif
