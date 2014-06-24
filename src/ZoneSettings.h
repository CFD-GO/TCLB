#include "Consts.h"
#include "cross.h"
#include <stdlib.h>
#include <assert.h>

#ifndef ZONESETTINGS_H
#define ZONESETTINGS_H

class ZoneSettings {
  int len;
  real_t * gpuValues;
  real_t * gpuControl;
  real_t ** cpuTab;
  real_t * cpuConst;
  real_t * cpuValues;
  real_t * cpuControl;

public:
  real_t ** gpuTab;
  real_t * gpuConst;
    
  inline ZoneSettings() {
    cpuTab = (real_t**) malloc(sizeof(real_t*) * ZONE_MAX * ZONESETTINGS);
    for (int i=0; i<ZONE_MAX * ZONESETTINGS; i++) {
      cpuTab[i] = NULL;
    }
    CudaMalloc(&gpuTab, sizeof(real_t*) * ZONE_MAX * ZONESETTINGS);
    cpuConst = (real_t*) malloc(sizeof(real_t) * ZONE_MAX * ZONESETTINGS);
    for (int i=0; i<ZONE_MAX * ZONESETTINGS; i++) {
      cpuConst[i] = 1.01;
    }
    for (int i=0; i<ZONE_MAX; i++) {
      for (int j=0; j<ZONESETTINGS; j++) {
        cpuConst[j+ZONESETTINGS*i] = 0.01 * (i+1);
      }
    }
    CudaMalloc(&gpuTab,   sizeof(real_t*) * ZONE_MAX * ZONESETTINGS);
    CudaMalloc(&gpuConst, sizeof(real_t)  * ZONE_MAX * ZONESETTINGS);
    CopyToGPU();
  }

  inline void set(int s, int z, real_t val) {
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
  
  inline void CopyToGPU () {
    CudaMemcpy(gpuTab,   cpuTab,   sizeof(real_t*) * ZONE_MAX * ZONESETTINGS, cudaMemcpyHostToDevice);
    CudaMemcpy(gpuConst, cpuConst, sizeof(real_t)  * ZONE_MAX * ZONESETTINGS, cudaMemcpyHostToDevice);
  }

  inline ~ZoneSettings() {
    free(cpuTab);
    CudaFree(&gpuTab);
    free(cpuConst);
    CudaFree(&gpuConst);
  }
};

#endif