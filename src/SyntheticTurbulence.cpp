#include "SyntheticTurbulence.h"
#include "cross.h"
#include <mpi.h>
#include <assert.h>
#include <stdlib.h>

void SyntheticTurbulence::CopyToGPU(STWaveSet & ST) {
 ST.resize(cpuset.nmodes, ST_GPU);
 ST.TimeWN = cpuset.TimeWN;
 if (ST.nmodes != 0) {
  CudaMemcpy(ST.data, cpuset.data, sizeof(real_t) * ST.nmodes * ST_DATA, cudaMemcpyHostToDevice);
 }
}

void runif(int n, double * tab) {
 for (int i=0;i<n;i++)
   tab[i] = (double)rand()/RAND_MAX;
}

void rnorm(int n, double *tab) {
  for (int i=0;i<n;i++) {
   double w[2];
   runif(2, w);
   w[0] = w[0] * atan(1.0) * 8;
   w[1] = sqrt(-log(w[1]));
   tab[i] = cos(w[0])*w[1];
   i++;
   if (i >= n) break;
   tab[i] = sin(w[0])*w[1];
  }
}

double skal(int n, double * w, double * v) {
 double ret=0;
 for (int i=0;i<n;i++) ret += w[i]*v[i];
 return ret;
}
 
SyntheticTurbulence::SyntheticTurbulence() {
 size = 0;
 cpuset.setsize(0, ST_CPU);
 Amplitudes = NULL;
 WaveLengths = NULL; 
 TimeWN = 0;
 cpuset.TimeWN = 0;
}
 

void SyntheticTurbulence::Generate() {
 double tab[6];
 assert(size == cpuset.nmodes);
 for (int j=0;j<size;j++) {
  if (D_MPI_RANK == 0) {
   rnorm(6,tab);
   double l;
   l = sqrt(skal(3,tab,tab));
   for (int i=0;i<3;i++) tab[i] /= l;
   l = skal(3,tab,tab+3);
   for (int i=0;i<3;i++) tab[i+3] -= tab[i]*l;
   l = sqrt(skal(3,tab+3,tab+3));
   for (int i=0;i<3;i++) tab[i+3] *= Amplitudes[j]/l;
   for (int i=0;i<6;i++) 
    cpuset.data[j*ST_DATA+i] = tab[i];
   cpuset.data[j*ST_DATA+ST_WAVE_L] = WaveLengths[j];
  }
  MPI_Bcast( cpuset.data, ST_DATA*size, MPI_REAL_T, 0, MPI_COMM_WORLD);
 }
  
}

void SyntheticTurbulence::resize(int n) {
 if (size != n) {
  if (Amplitudes != NULL) free(Amplitudes);
  if (WaveLengths != NULL) free(WaveLengths);
  cpuset.resize(n, ST_CPU);
  size = n;
  if (size == 0) {
   Amplitudes = NULL;
   WaveLengths = NULL;
  } else {
   Amplitudes = (double *) malloc(size*sizeof(double));
   WaveLengths = (double *) malloc(size*sizeof(double));
  }
 }
}

void SyntheticTurbulence::CalcEven() {

}

void SyntheticTurbulence::CalcQuant() {

}

double SyntheticTurbulence::EnergySpectrum(double w) {
 return 0;
}

void SyntheticTurbulence::setVonKarman(double Le, double Ld, double Lmin, double Lmax){

 assert(size == cpuset.nmodes);
 double dL = (Lmax-Lmin)/size;
 for (int i=0;i<size;i++) {
  double L;
  L = i*dL + dL/2 + Lmin;

  WaveLengths[i] = L;
  double c = 0.9685081;
  double E = c / Le * pow(L/Le,4.0) / pow(1.0+pow(L/Le,2.0),17./6.)*exp(-2.0*pow(L/Ld,2.0));
  Amplitudes[i] = sqrt(E * dL);
  printf(" %lg -> %lg (%lg)\n", WaveLengths[i], Amplitudes[i], E);
 }
 double sum=0;
 for (int i=0;i<size;i++) sum = sum + pow(Amplitudes[i],2.0);
 output("Total energy of synthetic turbulence: %2.0lf%% of spectrum\n", sum*100);
 if (sum < 0.7) NOTICE("Total energy of synthetic turbulence is below 70%% of the spectrum\n");
 else if (sum < 0.8) notice("Total energy of synthetic turbulence is below 80%% of the spectrum\n");
 else if (sum > 1) NOTICE("Total energy of synthetic turbulence is above 100%% of the spectrum\n");
 
 Generate();
}

void SyntheticTurbulence::setOneWave(double L){
 resize(1);
 Amplitudes[0] = 1;
 WaveLengths[0] = L;
 Generate();
}

void SyntheticTurbulence::setTimeScale(double L){
 TimeWN = L;
 cpuset.TimeWN = L;
}
    
