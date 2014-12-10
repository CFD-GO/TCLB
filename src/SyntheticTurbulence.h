#ifndef H_SYNTHETICTURBULENCE
#define H_SYNTHETICTURBULENCE

#include "Consts.h"
#include "Global.h"
#include "types.h"
#include <math.h>
#include <stdlib.h>
#define ST_DATA 7
#define ST_WAVE_X 0
#define ST_WAVE_Y 1
#define ST_WAVE_Z 2
#define ST_SINE_X 3
#define ST_SINE_Y 4
#define ST_SINE_Z 5
#define ST_WAVE_L 6
#define ST_GPU 0
#define ST_CPU 1

struct STWaveSet {
 int nmodes;
 real_t * data;

 void setsize(int n, int type) {
  nmodes = n;
  switch (type) {
  case ST_GPU:
   CudaMalloc(&data, nmodes*ST_DATA*sizeof(real_t));
   break;
  case ST_CPU:
   data = (real_t*) malloc(nmodes*ST_DATA*sizeof(real_t));
   break;
  }
 }
 
 void free_data(int type) {
  switch (type) {
  case ST_GPU:
   CudaFree(data);
   break;
  case ST_CPU:
   free(data);
   break;
  }
 }
 void resize(int n, int type) {
  if (n != nmodes) {
   free_data(type);
   setsize(n,type);
  }
 }
};

 enum eSpec {
  NoneSpec,
  vonKarmanSpec
 };


class SyntheticTurbulence {
private:
 int size;
 STWaveSet cpuset;
 eSpec SelectedSpec;
 double MaxEnWaveLen, DissWaveLen;
 double *WaveLengths, *Amplitudes;
public:
 SyntheticTurbulence();
 void CopyToGPU(STWaveSet & ST);
 void Generate();
 void CalcEven();
 void CalcQuant();
 double EnergySpectrum(double w);
 void SetVonKarman(double Le, double Ld); 
 void resize(int n);
};


inline CudaDeviceFunction vector_t calc(const STWaveSet &ST, real_t x, real_t y, real_t z) {
  vector_t ret;
  ret.x=0;ret.y=0;ret.z=0;
  for (int i=0; i<ST.nmodes; i++) {
    real_t x1 = ST.data[i*ST_DATA+ST_WAVE_X];
    real_t y1 = ST.data[i*ST_DATA+ST_WAVE_Y];
    real_t z1 = ST.data[i*ST_DATA+ST_WAVE_Z];
    real_t x2 = ST.data[i*ST_DATA+ST_SINE_X];
    real_t y2 = ST.data[i*ST_DATA+ST_SINE_Y];
    real_t z2 = ST.data[i*ST_DATA+ST_SINE_Z];
    real_t w = ST.data[i*ST_DATA+ST_WAVE_L];
    w = (x1*x + y1*y + z1*z) * w;
    real_t sw = sin(w), cw = cos(w);
    ret.x = sw*x2 + cw*(y1*z2-z1*y2);
    ret.y = sw*y2 + cw*(z1*x2-x1*z2);
    ret.z = sw*z2 + cw*(x1*y2-y1*x2);
  }
  return ret;
}


#endif