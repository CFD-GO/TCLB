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
 real_t TimeWN;
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

 enum eSpread {
  EvenSpread,
  LogSpread,
  QuantileSpread
 };


class SyntheticTurbulence {
private:
 int size;
 STWaveSet cpuset;
 double TimeWN;
 eSpec SelectedSpec;
 double MaxEnWaveLen, DissWaveLen;
 double *WaveLengths, *Amplitudes;
 eSpread spread;
public:
 SyntheticTurbulence();
 void CopyToGPU(STWaveSet & ST);
 void Generate();
 void CalcEven();
 void CalcQuant();
 double EnergySpectrum(double w);
 void setVonKarman(double Le, double Ld, double Lmin, double Lmax); 
 void setOneWave(double L); 
 void setTimeScale(double L); 
 inline void setSpread(eSpread s) {spread = s;}
 void resize(int n);
};

#endif