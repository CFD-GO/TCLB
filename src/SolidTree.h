#ifndef BALLTREE_H

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <time.h>
#include <vector>
#include <set>
#include <math.h>

typedef char tr_flag_t;
typedef int tr_addr_t;
typedef double tr_real_t;

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
    struct finder_t {
        size_t data_size;
        tr_elem* data;
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
    void InitFinder(finder_t&);
    void CopyToGPU(finder_t&, CudaStream_t stream);
};

#define BALLTREE_H
#endif
