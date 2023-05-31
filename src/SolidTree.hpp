#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <time.h>
#include <vector>
#include <set>
#include <math.h>
#include "SolidTree.h"

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
void SolidTree<BALLS>::InitFinder (typename SolidTree<BALLS>::finder_t& finder) {
    data_size_max = 0;
	finder.data = NULL;
	finder.data_size = 0;
}

template <class BALLS>
void SolidTree<BALLS>::CleanFinder (typename SolidTree<BALLS>::finder_t& finder) {
    data_size_max = 0;
	if (finder.data != NULL) CudaFree(finder.data);
	finder.data_size = 0;
}


template <class BALLS>
void SolidTree<BALLS>::CopyToGPU (typename SolidTree<BALLS>::finder_t& finder, CudaStream_t stream) {
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
