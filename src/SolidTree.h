#ifndef SOLIDTREE_H

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <time.h>
#include <vector>
#include <set>
#include <math.h>
#include "types.h"

struct tr_elem {
    tr_flag_t flag;
    tr_addr_t right;
    tr_addr_t back;
    tr_real_t a;
    tr_real_t b;
};


template <class BALLS>
class SolidTree {
public:
    class finder_t;
    template <class T>
    class set_found_t {
        const finder_t& finder;
        real_t* particle_data;
        real_t lower[3];
        real_t upper[3];
        real_t point[3];
        class iterator_t {
            const set_found_t * set;
        	tr_addr_t nodei;
            tr_addr_t i;
	        CudaDeviceFunction inline bool valid_() { return nodei != -1; }
	        CudaDeviceFunction void go(bool go_left) {
                while (nodei != -1) {
                    tr_elem elem = set->finder.data[nodei];
                    if (elem.flag >= 4) { i = elem.right; break; }
                    int dir = elem.flag;
                    if (go_left) if (set->lower[dir] < elem.b) { nodei++; continue; }
                    go_left = true;
                    if (set->upper[dir] >= elem.a) { nodei = elem.right; continue; }
                    go_left = false;
                    nodei = elem.back;
                }    
	        }
            CudaDeviceFunction iterator_t(const set_found_t& set_) : set(&set_) {
                nodei = 0;
	            if (set->finder.data_size == 0) { nodei = -1; return;}
		        go(true);
            };
            CudaDeviceFunction iterator_t(): set(nullptr) { nodei = -1; return; };
            friend class set_found_t;
        public:
            CudaDeviceFunction T operator* () { return T(set->particle_data, i, set->point); }
            CudaDeviceFunction iterator_t& operator++() { 
           		if (nodei != -1) {
		            nodei = set->finder.data[nodei].back;
		            go(false);
		        }
	            return *this;
            }
            CudaDeviceFunction bool operator!=(const iterator_t& other) { return valid_();};
        };
        public:
            typedef iterator_t iterator;
            CudaDeviceFunction inline iterator begin() { return iterator(*this); };
            CudaDeviceFunction inline iterator end()   { return iterator(); };
            CudaDeviceFunction inline set_found_t(const finder_t& finder_, real_t* particle_data_, const real_t point_[], const real_t lower_[], const real_t upper_[]): finder(finder_),  particle_data(particle_data_) {
                for (int i=0;i<3;i++) { lower[i]=lower_[i]; upper[i]=upper_[i]; point[i]=point_[i]; }
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
                tr_addr_t nodei = 0;
                bool go_left = true;
                cache_size = 0;
                while (nodei != -1) {
                    tr_elem elem = finder.data[nodei];
                    if (elem.flag >= 4) {
                        cache[cache_size] = elem.right;
                        ++cache_size;
                        if (cache_size < MAX_CACHE) {
                            nodei = elem.back;
                            go_left = false;
                            continue;
                        }
                        break;
                    }
                    int dir = elem.flag;
                    if (go_left) if (lower[dir] < elem.b) { nodei++; continue; }
                    go_left = true;
                    if (upper[dir] >= elem.a) { nodei = elem.right; continue; }
                    go_left = false;
                    nodei = elem.back;
                }    
            };
    };
    class finder_t {
        size_t data_size;
        tr_elem* data;
        friend class SolidTree<BALLS>;
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
    std::vector<tr_elem> tree;
    int half (int i, int j, int dir, tr_real_t thr);
    tr_addr_t build (int ind, int n, int back);
    std::vector<tr_addr_t> nr;
    size_t data_size_max;
public:
    BALLS* balls;
    inline void Build() {
        tree.clear();
        size_t n = balls->size();
        nr.resize(n);
        if (n > 0) {
            for (size_t i=0; i<n; ++i) nr[i] = i;
            build(0,n,-1);
        }
    }
    inline void InitFinder(finder_t&);
    inline void CleanFinder(finder_t&);
    inline void CopyToGPU(finder_t&, CudaStream_t stream);
};

template <class BALLS>
int SolidTree<BALLS>::half (int i, int j, int dir, tr_real_t thr) {
    if (i == (--j)) return i;
    while (true) {
        while (balls->getPos(nr[i],dir) <= thr) if ((++i) == j) return i;
        while (balls->getPos(nr[j],dir) >  thr) if (i == (--j)) return i;
        int tmp = nr[i];
        nr[i] = nr[j];
        nr[j] = tmp;
        if ((++i) == j) return i;
        if (i == (--j)) return i;
    }
}

template <class BALLS>
tr_addr_t SolidTree<BALLS>::build (int ind, int n, int back) {
    //    printf("tree build(%d %d %d)\n", ind, n, back);
    int node = tree.size();
    tr_elem elem;
    tree.push_back(elem);
    elem.back = back;
    if (n-ind < 2) {
        elem.flag = 4;
        elem.right = nr[ind];
    } else {
        tr_real_t sum=0.0;
        tr_real_t max_span=-1;
        int dir = 0;
        for (int ndir =0; ndir < 3; ndir++) {
            tr_real_t nsum = 0;
            tr_real_t val = balls->getPos(nr[ind],ndir);
            tr_real_t v_min = val, v_max = val;
            for (int i=ind; i<n; i++) {
                    val = balls->getPos(nr[i],ndir);
                    nsum += val;
                    if (val > v_max) v_max = val;
                    if (val < v_min) v_min = val;
            }
            if (v_max - v_min > max_span) {
                    max_span = v_max - v_min;
                    dir = ndir;
                    sum = nsum;
            }
        }
        sum /= (n-ind);
        tr_real_t v_max, v_min, v0, v1;
        //        printf("in dir %d: %lg %lg \n", dir, sum, max_span);
        int d = half(ind, n, dir, sum);
        if (!(ind < d)) {
            printf("Something is wrong in ball tree build: %d = half(%d, %d, %d, %lg); // max_span = %lg\n", d, ind, n, dir, sum, max_span);
        }
        assert(ind<d);
        assert(d<n);
        {
            v1 = v_max = balls->getPos(nr[ind],dir) + balls->getRad(nr[ind]);
            v0 = v_min = balls->getPos(nr[n-1],dir) - balls->getRad(nr[n-1]);
            for (int i=ind; i<n; i++) {
                    tr_real_t vala = balls->getPos(nr[i],dir) + balls->getRad(nr[i]);
                    tr_real_t valb = balls->getPos(nr[i],dir) - balls->getRad(nr[i]);
                    if (i < d) {
                        if (vala > v_max) v_max = vala;
                    } else {
                        if (valb < v_min) v_min = valb;
                    }
            }
        }
        //        printf("%d %3d %3d %lg %lg %lg %d --- %lg (%2.0lf%% %2.0lf%%)\n", node, ind, n, sum, v_min, v_max, d, v_max-v_min, 100.0*(d-ind)/(n-ind),100.0*(n-d)/(n-ind));
        build(ind, d, node);
        elem.right = build(d, n, back);
        elem.a = v_min;
        elem.b = v_max;
        elem.flag = dir;
    }
    tree[node] = elem;
    return node;
}


template <class BALLS>
void SolidTree<BALLS>::InitFinder (finder_t& finder) {
    data_size_max = 0;
    finder.data = NULL;
    finder.data_size = 0;
}

template <class BALLS>
void SolidTree<BALLS>::CleanFinder (finder_t& finder) {
    data_size_max = 0;
    if (finder.data != NULL) CudaFree(finder.data);
    finder.data_size = 0;
}


template <class BALLS>
void SolidTree<BALLS>::CopyToGPU (finder_t& finder, CudaStream_t stream) {
    if (tree.size() > data_size_max) {
        if (finder.data != NULL) CudaFree(finder.data);
        data_size_max = tree.size();
        CudaMalloc(&finder.data, tree.size() * sizeof(tr_elem));
    }
    finder.data_size = tree.size();
    if (tree.size() > 0) {
        CudaMemcpyAsync(finder.data, (tr_elem*) &tree[0], tree.size() * sizeof(tr_elem), CudaMemcpyHostToDevice, stream);
    }
}

#define SOLIDTREE_H
#endif
