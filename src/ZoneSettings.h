#include "Consts.h"
#include "cross.h"

class ZoneSettings () {
  int len;
  real_t ** gpuTab;
  real_t * gpuValues;
  real_t * gpuControl;
  real_t ** cpuTab;
  real_t * cpuValues;
  real_t * cpuControl;
  
  inline ZoneSettings() {
    cpuTab = (real_t**) malloc(sizeof(real_t*) * ZONE_MAX * ZONESETTINGS);
    for (int i=0; i<ZONE_MAX * ZONESETTINGS; i++) {
      cpuTab[i] = NULL;
    }
    CudaMalloc(&gpuTab, sizeof(real_t*) * ZONE_MAX * ZONESETTINGS);
  }

  inline ~ZoneSettings() {
    free(cpuTab);
    CudaFree(&gpuTab);
  }
}