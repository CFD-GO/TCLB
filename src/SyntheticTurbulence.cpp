#include "SyntheticTurbulence.h"
#include "cross.h"
#include <mpi.h>
#include <assert.h>
#include <stdlib.h>

void SyntheticTurbulence::CopyToGPU(STWaveSet & ST) {
 ST.resize(cpuset.nmodes, ST_GPU);
 if (ST.nmodes != 0) {
  CudaMemcpy(ST.data, cpuset.data, sizeof(real_t) * ST.nmodes * ST_DATA, cudaMemcpyHostToDevice);
 }
}

void runif(int n, double * tab) {
 for (int i=0;i<n;i++)
   tab[i] = (double)rand()/RAND_MAX;
}

double rnorm(int n, double *tab) {
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
 resize(1);
 Amplitudes[0] = 1;
 WaveLengths[0] = 1./20;
 Generate();
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

}

void SyntheticTurbulence::SetVonKarman(double Le, double Ld){

}
