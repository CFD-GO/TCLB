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
    cpuValues = (real_t**) malloc(sizeof(real_t*) * ZONE_MAX * ZONESETTINGS);
    cpuTab = (real_t**) malloc(sizeof(real_t*) * ZONE_MAX * ZONESETTINGS);
    for (int i=0; i<ZONE_MAX * ZONESETTINGS; i++) {
      cpuValues[i] = NULL;
      cpuTab[i] = NULL;
    }
    CudaMalloc(&gpuTab, sizeof(real_t*) * ZONE_MAX * ZONESETTINGS);
    cpuConst = (real_t*) malloc(sizeof(real_t) * ZONE_MAX * ZONESETTINGS);
    for (int i=0; i<ZONE_MAX * ZONESETTINGS; i++) {
      cpuConst[i] = 0.0;
    }
    CudaMalloc(&gpuTab,   sizeof(real_t*) * ZONE_MAX * ZONESETTINGS);
    CudaMalloc(&gpuConst, sizeof(real_t)  * ZONE_MAX * ZONESETTINGS);
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
      if (cpuValues[i] != NULL) free(cpuValues[i]);
      if (cpuTab[i] != NULL) CudaFree(cpuTab[i]);
    }
    len = nlen;
    CopyToGPU();  
  }    

  inline void Alloc(int i) {
    assert(cpuValues[i] == NULL);
    cpuValues[i] = (real_t*) malloc(sizeof(real_t) * len);
    CudaMalloc(&cpuTab[i], sizeof(real_t) * len);
  }    

  inline void set(int s, int z, std::vector<double> val) {
    assert(s >=  0);
    assert(s <   ZONESETTINGS);
    assert(z >= 0);
    assert(z <   ZONE_MAX);
    assert(val.size() == len);
    int i = s+ZONESETTINGS*z;
    Alloc(i);
    for (int j=0; j<len; j++) {
      cpuValues[i][j] = val[j];
    }
    CopyToGPU();  
  }
  

  
  inline void CopyToGPU () {
    CudaMemcpy(gpuTab,   cpuTab,   sizeof(real_t*) * ZONE_MAX * ZONESETTINGS, cudaMemcpyHostToDevice);
    CudaMemcpy(gpuConst, cpuConst, sizeof(real_t)  * ZONE_MAX * ZONESETTINGS, cudaMemcpyHostToDevice);
    for (int i=0; i<ZONE_MAX * ZONESETTINGS; i++) if (cpuValues[i] != NULL) {
      assert(cpuTab[i] != NULL);
      CudaMemcpy(cpuTab[i],   cpuValues[i],  sizeof(real_t) * len, cudaMemcpyHostToDevice);
    }
  }

  inline ~ZoneSettings() {
    for (int i=0; i<ZONE_MAX * ZONESETTINGS; i++) {
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