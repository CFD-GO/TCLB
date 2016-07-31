#include "Consts.h"
#include "cross.h"
#include <stdlib.h>
#include <assert.h>

#ifndef ZONESETTINGS_H
#define ZONESETTINGS_H

class ZoneSettings {
  size_t len;
  real_t ** cpuTab;
  real_t * cpuConst;
public:
  int MaxZones;
  real_t ** cpuValues;
  real_t ** gpuTab;
  real_t * gpuConst;
    
  inline ZoneSettings() {
    DEBUG_M;
    len = 1;
    MaxZones=0;
    debug1("TIME_SEG: %d\n", TIME_SEG);
    cpuValues = (real_t**) malloc(sizeof(real_t*) * TIME_SEG);
    assert(cpuValues != NULL);
    cpuTab = (real_t**) malloc(sizeof(real_t*) * TIME_SEG);
    assert(cpuTab != NULL);
    for (int i=0; i<TIME_SEG; i++) {
      cpuValues[i] = NULL;
      cpuTab[i] = NULL;
    }
    cpuConst = (real_t*) malloc(sizeof(real_t) * TIME_SEG);
    assert(cpuConst != NULL);
    for (int i=0; i<TIME_SEG; i++) {
      cpuConst[i] = 0.0;
    }
    DEBUG_M;
    debug0("&gpuTab: %p, size: %ld\n", &gpuTab, sizeof(real_t*) * TIME_SEG);
    CudaMalloc((void**) &gpuTab, sizeof(real_t*) * TIME_SEG);
    assert(gpuTab != NULL);
    DEBUG_M;
    CudaMalloc((void**) &gpuConst,   sizeof(real_t)  * TIME_SEG);
    assert(gpuConst != NULL);
    DEBUG_M;
    CopyToGPU();
    DEBUG_M;
  }
  
  inline void zone_max(int z) {
    if (z+1 > MaxZones) {
      MaxZones = z+1;
      output("Setting number of zones to %d\n", MaxZones);
    }
  }

  inline void set(int s, int z, double val) {
    assert(s >=  0);
    assert(s <   ZONESETTINGS);
    assert(z >= -1);
    assert(z <   ZONE_MAX);
    zone_max(z);
    if (z == -1) {
      for (int i=0;i<ZONE_MAX; i++) {
        if (cpuConst[s+ZONESETTINGS*i] != 0.0 && cpuConst[s+ZONESETTINGS*i] != val ){
             WARNING("Zone-specific settings in zone %d were overwritten from %lf to %lf", i, cpuConst[s+ZONESETTINGS*i], val );
        }
        cpuConst[s+ZONESETTINGS*i] = val;
      }
    } else {
      cpuConst[s+ZONESETTINGS*z] = val;
    }
    CopyToGPU();  
  }
  
  inline void setLen(size_t nlen) {
    for (int i=0; i<TIME_SEG; i++) {
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
    if (cpuValues[i] == NULL) {
      cpuValues[i] = (real_t*) malloc(sizeof(real_t) * len);
      CudaMalloc(&cpuTab[i], sizeof(real_t) * len);
    }
  }    

  inline void set_internal(int i, std::vector<double> val) {
    assert(val.size() == len);
    set_internal(i, &val[0]);
  }

  inline void set_internal(int i, const double* val) {
    Alloc(i);
    for (size_t j=0; j<len; j++) {
      cpuValues[i][j] = val[j];
    }
    Alloc(i+DT_OFFSET);
    for (size_t j=1; j<len-1; j++) {
      cpuValues[i+DT_OFFSET][j] = (val[j+1] - val[j-1])/2;
    }
    cpuValues[i+DT_OFFSET][0] = (val[1] - val[len-1])/2;
    cpuValues[i+DT_OFFSET][len-1] = (val[0] - val[len-2])/2;
    Alloc(i+GRAD_OFFSET);
    Alloc(i+GRAD_OFFSET+DT_OFFSET);
    for (size_t j=0; j<len; j++) {
      cpuValues[i+GRAD_OFFSET][j] = 0;
      cpuValues[i+GRAD_OFFSET+DT_OFFSET][j] = 0;
    }
  }


  inline void set(int s, int z, std::vector<double> val) {
    assert(s >=  0);
    assert(s <   ZONESETTINGS);
    assert(z >= -1);
    assert(z <   ZONE_MAX);
    assert(val.size() == len);
    zone_max(z);
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

  inline void set(int s, int z, const double* val) {
    assert(s >=  0);
    assert(s <   ZONESETTINGS);
    assert(z >= -1);
    assert(z <   ZONE_MAX);
    zone_max(z);
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
  
  inline double get(int s, int z, size_t it) {
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


  inline void get(int s, int z, double * tab) {
    assert(s >=  0);
    assert(s <   ZONESETTINGS);
    assert(z >=  0);
    assert(z <   ZONE_MAX);
    int i = s+ZONESETTINGS*z;
    if (cpuValues[i] == NULL) {
      tab[0] = cpuConst[i];
    } else {
      for (size_t it=0; it<len; it++) {
        tab[it] = cpuValues[i][it];
      }
    }
  }

  inline void get_grad(int s, int z, double * tab) {
    assert(s >=  0);
    assert(s <   ZONESETTINGS);
    assert(z >=  0);
    assert(z <   ZONE_MAX);
    CopyFromGPU();
    int i = s + ZONESETTINGS*z + GRAD_OFFSET;
    if (cpuValues[i] == NULL) {
      tab[0] = cpuConst[i];
    } else {
      for (size_t it=0; it<len; it++) {
        tab[it] = cpuValues[i][it];
      }
      for (size_t j=1; j<len-1; j++) {
        tab[j+1] += 0.5*cpuValues[i+DT_OFFSET][j];
        tab[j-1] -= 0.5*cpuValues[i+DT_OFFSET][j];
      }
        tab[1] += 0.5*cpuValues[i+DT_OFFSET][0];
        tab[len-1] -= 0.5*cpuValues[i+DT_OFFSET][0];
        tab[0] += 0.5*cpuValues[i+DT_OFFSET][len-1];
        tab[len-2] -= 0.5*cpuValues[i+DT_OFFSET][len-1];
    }
  }

  inline size_t getLen(int s, int z) {
    assert(s >=  0);
    assert(s <   ZONESETTINGS);
    assert(z >=  0);
    assert(z <   ZONE_MAX);
    int i = s+ZONESETTINGS*z;
    if (cpuValues[i] == NULL) {
      return 1;
    } else {
      return len;
    }
  }

  
  inline void CopyToGPU () {
    DEBUG_M;
    CudaMemcpy(gpuTab,   cpuTab,   sizeof(real_t*) * TIME_SEG, cudaMemcpyHostToDevice);
    CudaMemcpy(gpuConst, cpuConst, sizeof(real_t)  * GRAD_OFFSET, cudaMemcpyHostToDevice);
    DEBUG_M;
    for (int i=0; i<GRAD_OFFSET; i++) if (cpuValues[i] != NULL) {
      assert(cpuTab[i] != NULL);
      CudaMemcpy(cpuTab[i],   cpuValues[i],  sizeof(real_t) * len, cudaMemcpyHostToDevice);
    }
  }

  inline void ClearGrad () {
    for (int i=GRAD_OFFSET; i<TIME_SEG; i++) if (cpuValues[i] != NULL) {
      debug0("Clearing gradient in ZoneSettings (%d)\n", i);
      CudaMemset(cpuTab[i], 0, sizeof(real_t) * len);
    }
  }

  inline void CopyFromGPU () {
    DEBUG_M;
    for (int i=GRAD_OFFSET; i<TIME_SEG; i++) if (cpuValues[i] != NULL) {
      assert(cpuTab[i] != NULL);
      debug0("Copying gradient data from GPU (%d)\n", i);
      CudaMemcpy(cpuValues[i], cpuTab[i],  sizeof(real_t) * len, cudaMemcpyDeviceToHost);
    }
  }

  inline ~ZoneSettings() {
    for (int i=0; i<TIME_SEG; i++) {
      if (cpuValues[i] != NULL) free(cpuValues[i]);
      if (cpuTab[i] != NULL) CudaFree(cpuTab[i]);
    }
    free(cpuTab);
    CudaFree(gpuTab);
    free(cpuConst);
    CudaFree(gpuConst);
  }
  
  inline size_t getLen() { return len; }
};

#endif
