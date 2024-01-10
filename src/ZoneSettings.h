#include "Consts.h"
#include "Global.h"
#include "cross.h"
#include "types.h"
#include <stdlib.h>
#include <assert.h>
#include <vector>

#ifndef ZONESETTINGS_H
#define ZONESETTINGS_H

class ZoneSettings {
  size_t len;
  real_t ** cpuTab;
  real_t * cpuConst;
  const int zonesettings;
  const int zones;
  inline int dt_offset() { return zones * zonesettings; }
  inline int time_seg() { return 4 * zones * zonesettings; }
  inline int grad_offset() { return  2 * zones * zonesettings; }
public:
  int MaxZones;
  real_t ** cpuValues;
  real_t ** gpuTab;
  real_t * gpuConst;
    
  inline ZoneSettings(int zonesettings_, int zones_) : zonesettings(zonesettings_), zones(zones_) {
    DEBUG_M;
    len = 1;
    MaxZones=0;
    debug1("time_seg(): %d\n", time_seg());
    cpuValues = (real_t**) malloc(sizeof(real_t*) * time_seg());
    assert(time_seg() == 0 || cpuValues != NULL);
    cpuTab = (real_t**) malloc(sizeof(real_t*) * time_seg());
    assert(time_seg() == 0 || cpuTab != NULL);
    for (int i=0; i<time_seg(); i++) {
      cpuValues[i] = NULL;
      cpuTab[i] = NULL;
    }
    cpuConst = (real_t*) malloc(sizeof(real_t) * time_seg());
    assert(time_seg() == 0 || cpuConst != NULL);
    for (int i=0; i<time_seg(); i++) {
      cpuConst[i] = 0.0;
    }
    DEBUG_M;
    debug0("&gpuTab: %p, size: %ld\n", &gpuTab, sizeof(real_t*) * time_seg());
    CudaMalloc((void**) &gpuTab, sizeof(real_t*) * time_seg());
    assert(time_seg() == 0 || gpuTab != NULL);
    DEBUG_M;
    CudaMalloc((void**) &gpuConst,   sizeof(real_t)  * time_seg());
    assert(time_seg() == 0 || gpuConst != NULL);
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
    assert(s <   zonesettings);
    assert(z >= -1);
    assert(z <   zones);
    zone_max(z);
    bool is_already_reported = false;
    if (z == -1) {
      for (int i=0;i<zones; i++) {
        if (cpuConst[s+zonesettings*i] != 0.0 && cpuConst[s+zonesettings*i] != val && !is_already_reported){
             WARNING("Zone-specific settings in zone %d were overwritten from %lf to %lf", i, cpuConst[s+zonesettings*i], val );
             is_already_reported = true;
        }
        cpuConst[s+zonesettings*i] = val;
      }
    } else {
      cpuConst[s+zonesettings*z] = val;
    }
    CopyToGPU();  
  }
  
  inline void setLen(size_t nlen) {
    for (int i=0; i<time_seg(); i++) {
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
    Alloc(i+dt_offset());
    for (size_t j=1; j<len-1; j++) {
      cpuValues[i+dt_offset()][j] = (val[j+1] - val[j-1])/2;
    }
    cpuValues[i+dt_offset()][0] = (val[1] - val[len-1])/2;
    cpuValues[i+dt_offset()][len-1] = (val[0] - val[len-2])/2;
    Alloc(i+grad_offset());
    Alloc(i+grad_offset()+dt_offset());
    for (size_t j=0; j<len; j++) {
      cpuValues[i+grad_offset()][j] = 0;
      cpuValues[i+grad_offset()+dt_offset()][j] = 0;
    }
  }


  inline void set(int s, int z, std::vector<double> val) {
    assert(s >=  0);
    assert(s <   zonesettings);
    assert(z >= -1);
    assert(z <   zones);
    assert(val.size() == len);
    zone_max(z);
    if (z == -1) {
      for (int z=0;z<zones; z++) {
        int i = s+zonesettings*z;
        set_internal(i,val);
      }
    } else {
      int i = s+zonesettings*z;
      set_internal(i,val);
    }
    CopyToGPU();  
  }

  inline void set(int s, int z, const double* val) {
    assert(s >=  0);
    assert(s <   zonesettings);
    assert(z >= -1);
    assert(z <   zones);
    zone_max(z);
    if (z == -1) {
      for (int z=0;z<zones; z++) {
        int i = s+zonesettings*z;
        set_internal(i,val);
      }
    } else {
      int i = s+zonesettings*z;
      set_internal(i,val);
    }
    CopyToGPU();  
  }
  
  inline double get(int s, int z, size_t it) {
    assert(s >=  0);
    assert(s <   zonesettings);
    assert(z >=  0);
    assert(z <   zones);
    int i = s+zonesettings*z;
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
    assert(s <   zonesettings);
    assert(z >=  0);
    assert(z <   zones);
    int i = s+zonesettings*z;
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
    assert(s <   zonesettings);
    assert(z >=  0);
    assert(z <   zones);
    CopyFromGPU();
    int i = s + zonesettings*z + grad_offset();
    if (cpuValues[i] == NULL) {
      tab[0] = cpuConst[i];
    } else {
      for (size_t it=0; it<len; it++) {
        tab[it] = cpuValues[i][it];
      }
      for (size_t j=1; j<len-1; j++) {
        tab[j+1] += 0.5*cpuValues[i+dt_offset()][j];
        tab[j-1] -= 0.5*cpuValues[i+dt_offset()][j];
      }
        tab[1] += 0.5*cpuValues[i+dt_offset()][0];
        tab[len-1] -= 0.5*cpuValues[i+dt_offset()][0];
        tab[0] += 0.5*cpuValues[i+dt_offset()][len-1];
        tab[len-2] -= 0.5*cpuValues[i+dt_offset()][len-1];
    }
  }

  inline size_t getLen(int s, int z) {
    assert(s >=  0);
    assert(s <   zonesettings);
    assert(z >=  0);
    assert(z <   zones);
    int i = s+zonesettings*z;
    if (cpuValues[i] == NULL) {
      return 1;
    } else {
      return len;
    }
  }

  
  inline void CopyToGPU () {
    DEBUG_M;
    CudaMemcpy(gpuTab,   cpuTab,   sizeof(real_t*) * time_seg(), CudaMemcpyHostToDevice);
    CudaMemcpy(gpuConst, cpuConst, sizeof(real_t)  * grad_offset(), CudaMemcpyHostToDevice);
    DEBUG_M;
    for (int i=0; i<grad_offset(); i++) if (cpuValues[i] != NULL) {
      assert(cpuTab[i] != NULL);
      CudaMemcpy(cpuTab[i],   cpuValues[i],  sizeof(real_t) * len, CudaMemcpyHostToDevice);
    }
  }

  inline void ClearGrad () {
    for (int i=grad_offset(); i<time_seg(); i++) if (cpuValues[i] != NULL) {
      debug0("Clearing gradient in ZoneSettings (%d)\n", i);
      CudaMemset(cpuTab[i], 0, sizeof(real_t) * len);
    }
  }

  inline void CopyFromGPU () {
    DEBUG_M;
    for (int i=grad_offset(); i<time_seg(); i++) if (cpuValues[i] != NULL) {
      assert(cpuTab[i] != NULL);
      debug0("Copying gradient data from GPU (%d)\n", i);
      CudaMemcpy(cpuValues[i], cpuTab[i],  sizeof(real_t) * len, CudaMemcpyDeviceToHost);
    }
  }

  inline ~ZoneSettings() {
    for (int i=0; i<time_seg(); i++) {
      if (cpuValues[i] != NULL) free(cpuValues[i]);
      if (cpuTab[i] != NULL) CudaFree(cpuTab[i]);
    }
    free(cpuTab);
    CudaFree(gpuTab);
    free(cpuConst);
    CudaFree(gpuConst);
  }

  int dumpToFile(const char*);
  
  inline size_t getLen() { return len; }
};

#endif
