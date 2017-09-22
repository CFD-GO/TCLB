#ifndef REMOTEFORCEINTERFACE_H
#define REMOTEFORCEINTERFACE_H

#include "mpi.h" 
#include <stdlib.h>
#include <stdio.h>
#include <vector>

#include "TCLBForceGroupCommon.h"


class RemoteForceInterface {
public:
  RemoteForceInterface();
  ~RemoteForceInterface();
  int Start();
  inline const rfi_size_t size() const { return totsize; }
  inline rfi_real_t* Particles() { return &tab[0]; }
  void GetParticles();
  void SetParticles();
  void Close();
  inline bool Active() { return intercomm != MPI_COMM_NULL; }
private:
  int world_size, universe_size;
  int rank;
  int workers;
  int masters;
  MPI_Comm intercomm;
  rfi_size_t totsize;
  std::vector<rfi_real_t> tab;
  std::vector<rfi_size_t> sizes;
  std::vector<rfi_size_t> offsets;
  std::vector<MPI_Request> reqs;
  std::vector<MPI_Status> stats;
};

#endif
