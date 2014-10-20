#include "Consts.h"
#include "cross.h"
#include <stdlib.h>
#include <assert.h>

#ifndef ZONESETTINGS_H
#define ZONESETTINGS_H

class ZoneSettings {
  int len;
  real_t ** cpuTab;
  real_t * cpuConst;

public:
  real_t ** cpuValues;
  real_t ** gpuTab;
  real_t * gpuConst;
    
  inline ZoneSettings() {
    len = 1;
    cpuValues = (real_t**) malloc(2 * sizeof(real_t*) * ZONE_MAX * ZONESETTINGS);
    cpuTab = (real_t**) malloc(2 * sizeof(real_t*) * ZONE_MAX * ZONESETTINGS);
    for (int i=0; i<ZONE_MAX * ZONESETTINGS*2; i++) {
      cpuValues[i] = NULL;
      cpuTab[i] = NULL;
    }
    cpuConst = (real_t*) malloc(sizeof(real_t) * ZONE_MAX * ZONESETTINGS);
    for (int i=0; i<ZONE_MAX * ZONESETTINGS; i++) {
      cpuConst[i] = 0.0;
    }
    CudaMalloc(&gpuTab, 2 * sizeof(real_t*) * ZONE_MAX * ZONESETTINGS);
    CudaMalloc(&gpuConst,   sizeof(real_t)  * ZONE_MAX * ZONESETTINGS);
    CopyToGPU();
  }

  inline void set(int s, int z, double val) {
    assert(s >=  0);
    assert(s <   ZONESETTINGS);
    assert(z >= -1);
    assert(z <   ZONE_MAX);
    if (z == -1) {
      for (int i=0;i<ZONE_MAX; i++) {
        cpuConst[s+ZONESETTINGS*i] = val;
      }
    } else {
      cpuConst[s+ZONESETTINGS*z] = val;
    }
    CopyToGPU();  
  }
  
  inline void setLen(int nlen) {
    for (int i=0; i<ZONE_MAX * ZONESETTINGS; i++) {
      if (cpuValues[i] != NULL) {
         free(cpuValues[i]);
         cpuValues[i] = NULL;
      }
      if (cpuTab[i] != NULL) {
        CudaFree(cpuTab[i]);
        cpuTab[i] = NULL;
      }
    }
    len = nlen;
    CopyToGPU();  
  }    

  inline void Alloc(int i) {
    assert(cpuValues[i] == NULL);
    cpuValues[i] = (real_t*) malloc(sizeof(real_t) * len);
    CudaMalloc(&cpuTab[i], sizeof(real_t) * len);
  }    

  inline void set_internal(int i, std::vector<double> val) {
    Alloc(i);
    for (int j=0; j<len; j++) {
      cpuValues[i][j] = val[j];
    }
    Alloc(i+DT_OFFSET);
    for (int j=1; j<len-1; j++) {
      cpuValues[i+DT_OFFSET][j] = (val[j+1] - val[j-1])/2;
    }
    cpuValues[i+DT_OFFSET][0] = (val[1] - val[len-1])/2;
    cpuValues[i+DT_OFFSET][len-1] = (val[0] - val[len-2])/2;
  }


  inline void set(int s, int z, std::vector<double> val) {
    assert(s >=  0);
    assert(s <   ZONESETTINGS);
    assert(z >= -1);
    assert(z <   ZONE_MAX);
    assert(val.size() == len);
    if (z == -1) {
      for (int z=0;z<ZONE_MAX; z++) {
        int i = s+ZONESETTINGS*z;
        set_internal(i,val);
      }
    } else {
      int i = s+ZONESETTINGS*z;
      set_internal(i,val);
    }
    CopyToGPU();  
  }
  
  inline double get(int s, int z, int it) {
    assert(s >=  0);
    assert(s <   ZONESETTINGS);
    assert(z >=  0);
    assert(z <   ZONE_MAX);
    int i = s+ZONESETTINGS*z;
    if (cpuValues[i] == NULL) {
      return cpuConst[i];
    } else {
      assert(it >= 0);
      assert(it < len);
      return cpuValues[i][it];
    }
  }

  
  inline void CopyToGPU () {
    CudaMemcpy(gpuTab,   cpuTab,   sizeof(real_t*) * ZONE_MAX * ZONESETTINGS * 2, cudaMemcpyHostToDevice);
    CudaMemcpy(gpuConst, cpuConst, sizeof(real_t)  * ZONE_MAX * ZONESETTINGS, cudaMemcpyHostToDevice);
    for (int i=0; i<ZONE_MAX * ZONESETTINGS * 2; i++) if (cpuValues[i] != NULL) {
      assert(cpuTab[i] != NULL);
      CudaMemcpy(cpuTab[i],   cpuValues[i],  sizeof(real_t) * len, cudaMemcpyHostToDevice);
    }
  }

  inline ~ZoneSettings() {
    for (int i=0; i<ZONE_MAX * ZONESETTINGS * 2; i++) {
      if (cpuValues[i] != NULL) free(cpuValues[i]);
      if (cpuTab[i] != NULL) CudaFree(cpuTab[i]);
    }
    free(cpuTab);
    CudaFree(gpuTab);
    free(cpuConst);
    CudaFree(gpuConst);
  }
  
  inline int getLen() { return len; }
};

#endif