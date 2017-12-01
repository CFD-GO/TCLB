#ifndef REMOTEFORCEINTERFACE_H
#define REMOTEFORCEINTERFACE_H

#include "mpi.h" 
#include <stdlib.h>
#include <stdio.h>
#include <vector>

namespace rfi {

#define RFI_CODE_HANDSHAKE 1
#define RFI_CODE_FINISH 2
#define RFI_CODE_PARTICLES 3
#define RFI_CODE_FORCES 4
#define RFI_CODE_ABORT 0xFF

#define RFI_FINISHED ((size_t) -1)

#define RFI_DATA_R 0
#define RFI_DATA_POS 1
#define RFI_DATA_VEL 4
#define RFI_DATA_ANGVEL 7

#define RFI_DATA_VOL 0
#define RFI_DATA_FORCE 1
#define RFI_DATA_MOMENT 4

#define RFI_DATA_SIZE 20

#define MPI_SIZE_T MPI_UNSIGNED_LONG


enum rfi_type_t {
  ForceCalculator,
  ForceIntegrator
};
enum rfi_rot_t {
  RotParticle,
  NRotParticle
};
enum rfi_storage_t {
  ArrayOfStructures,
  StructureOfArrays
};


template < rfi_type_t TYPE, rfi_rot_t ROT, rfi_storage_t STORAGE = ArrayOfStructures, typename rfi_real_t = double >
class RemoteForceInterface {
private:
  int world_size; ///< Size of current program world
  int universe_size; ///< Size of the universe (both integrated programs)
  int rank; ///< My rank in world
  int workers; ///< Number of workers
  int masters; ///< Number of masters
  MPI_Comm intercomm; ///< Intercomm between master and slave
  size_t ntab; ///< Length of tab
  size_t totsize; ///< Total number of particles
  std::vector<rfi_real_t> tab; ///< Array storing all the data of particles
  std::vector<size_t> sizes; ///< Array of sizes of data recieved from each slave/master 
  std::vector<size_t> offsets; ///< Array of offsets of data recieved from each slave/master
  std::vector<MPI_Request> reqs; ///< Array of MPI requests for non-blocking calls
  std::vector<MPI_Status> stats; ///< Array of MPI status for non-blocking calls
  MPI_Datatype MPI_RFI_REAL_T; ///< The MPI datatype handle for rfi_real_t (either MPI_FLOAT or MPI_DOUBLE)
  MPI_Datatype MPI_PARTICLE; ///< The MPI datatype handle for rfi_real_t (either MPI_FLOAT or MPI_DOUBLE)
  bool rot;
  bool active;
  bool connected;
  int my_type;
  int Negotiate();
  void Zero();
  void Finish();
public:
  int particle_size;
  char * name;
  RemoteForceInterface();
  ~RemoteForceInterface();
  
  int Connect(MPI_Comm intercomm_);
  void Alloc();  
  inline const size_t size() const { return totsize; }
  inline const size_t mem_size() const { return ntab * sizeof(rfi_real_t); }
  inline rfi_real_t* Particles() { return &tab[0]; }
  void SendSizes();
  void SendParticles();
  void SendForces();
  void Close();
  inline bool Active() { return active; }
  inline bool Connected() { return connected; }
  inline int Workers() { return workers; }
  inline size_t& Size(int i) { return sizes[i]; }
  inline bool Rot() { return rot; }
  inline int space_for_workers() { return universe_size - world_size; };
  inline rfi_real_t& Data(size_t i, int j) {
    if (STORAGE == ArrayOfStructures) {
      return tab[i*particle_size + j];
    } else {
      return tab[i + j*totsize];
    }
  }
  inline rfi_real_t& getPos(size_t i, int j) {
    return Data(i, RFI_DATA_POS+j);
  }
  inline rfi_real_t& getRad(size_t i) {
    return Data(i, RFI_DATA_R);
  }
  inline void SetData(size_t i, int j, rfi_real_t val) {
    Data(i,j) = val;
  }
};

};
#endif
