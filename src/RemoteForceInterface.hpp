#include "RemoteForceInterface.h"

#ifndef debug1
 #define debug1(...)
 #define RFI_DEF_debug1
#endif
#ifndef ERROR
 #define ERROR printf
 #define RFI_DEF_ERROR
#endif
#ifndef output
 #define output printf
 #define RFI_DEF_output
#endif

namespace rfi {

template < rfi_type_t TYPE, rfi_rot_t ROT, rfi_storage_t STORAGE, typename rfi_real_t >
RemoteForceInterface < TYPE, ROT, STORAGE, rfi_real_t >::RemoteForceInterface() : workers(0), masters(0), intercomm(MPI_COMM_NULL), totsize(0), ntab(0) {
   connected = false;
   active = false;
   my_type = TYPE;
   if (TYPE == ForceIntegrator) {
     name = "ForceIntegrator";
   } else if (TYPE == ForceCalculator) {
     name = "ForceCalculator";
   } else {
     name = "N/A";
     ERROR("Unknown type of RemoteForceInterface");
     exit(-1);
   }
   particle_size = 0;
   if (sizeof(rfi_real_t) == 8)
    MPI_RFI_REAL_T = MPI_DOUBLE;
   else if (sizeof(rfi_real_t) == 4)
    MPI_RFI_REAL_T = MPI_FLOAT;
   else {
    ERROR("Unknown type rfi_real_t in RemoteForceInterface");
    exit(-1);
   }
}

template < rfi_type_t TYPE, rfi_rot_t ROT, rfi_storage_t STORAGE, typename rfi_real_t >
RemoteForceInterface < TYPE, ROT, STORAGE, rfi_real_t >::~RemoteForceInterface() {
  if (intercomm != MPI_COMM_NULL) {
    ERROR("RFI: This should never happen\n"); // LCOV_EXCL_LINE
  }
}

template < rfi_type_t TYPE, rfi_rot_t ROT, rfi_storage_t STORAGE, typename rfi_real_t >
int RemoteForceInterface < TYPE, ROT, STORAGE, rfi_real_t >::Negotiate() {
  if (! connected) return -1;
  output("RFI: %s: Starting negotiations ...\n", name);
  MPI_Barrier(intercomm);

  int my_rot = ROT == RotParticle;
  int other_rot;
  MPI_Allreduce( &my_rot, &other_rot, 1, MPI_INT, MPI_LAND, intercomm);
  rot = my_rot && other_rot;
  
  if (rot) {
   particle_size = 20;
   output("RFI: %s: Decided to calculate with rotation\n",name);
  } else {
   particle_size = 20;
   output("RFI: %s: Decided to calculate without rotation\n",name);
  }

  if (STORAGE == ArrayOfStructures) {
    MPI_Type_contiguous(particle_size, MPI_RFI_REAL_T, &MPI_PARTICLE);
  } else {
    MPI_Type_vector(particle_size, 1, totsize, MPI_RFI_REAL_T, &MPI_PARTICLE);
  }
  output("Adding type ...\n");
  MPI_Type_commit(&MPI_PARTICLE);

  output("RFI: %s: Finished negotiations\n",name);
  MPI_Barrier(intercomm);
  active = true;
  return 0;
}


template < rfi_type_t TYPE, rfi_rot_t ROT, rfi_storage_t STORAGE, typename rfi_real_t >
int RemoteForceInterface < TYPE, ROT, STORAGE, rfi_real_t >::Connect(MPI_Comm intercomm_) {
   if (connected) {
     ERROR("Already connected");
     return -1;
   }
   intercomm = intercomm_;
   MPI_Comm_remote_size(intercomm, &workers);
   MPI_Comm_size(intercomm, &masters);
   MPI_Comm_rank(intercomm, &rank);
   sizes.resize(workers, 0);
   offsets.resize(workers+1, 0);
   connected=true;
   Zero();
   return Negotiate();
}

template < rfi_type_t TYPE, rfi_rot_t ROT, rfi_storage_t STORAGE, typename rfi_real_t >
void RemoteForceInterface < TYPE, ROT, STORAGE, rfi_real_t >::Alloc() {
    offsets[0] = 0;
    for (int i=0; i<workers; i++) offsets[i+1] = offsets[i] + sizes[i];
    if (totsize != offsets[workers]) {
      totsize = offsets[workers];
      ntab = totsize * particle_size;
      if (ntab > tab.size()) tab.resize(ntab);
      if (STORAGE == ArrayOfStructures) {
      } else {
        MPI_Aint lb, extent;
        MPI_Type_get_extent(MPI_RFI_REAL_T, &lb, &extent);
        MPI_Datatype MPI_PARTICLE_;
        MPI_Type_vector(particle_size, 1, totsize, MPI_RFI_REAL_T, &MPI_PARTICLE_);
        MPI_Type_create_resized(MPI_PARTICLE_, lb, extent, &MPI_PARTICLE);
        output("Adding type MPI ...\n");
        MPI_Type_commit(&MPI_PARTICLE);
        MPI_Type_get_extent(MPI_PARTICLE, &lb, &extent);
      }
    }
}


template < rfi_type_t TYPE, rfi_rot_t ROT, rfi_storage_t STORAGE, typename rfi_real_t >
void RemoteForceInterface < TYPE, ROT, STORAGE, rfi_real_t >::Zero() {
  totsize = 0;
  for (int i=0; i<workers; i++) {
   sizes[i] = 0;
   offsets[i] = 0;
  }
}

template < rfi_type_t TYPE, rfi_rot_t ROT, rfi_storage_t STORAGE, typename rfi_real_t >
void RemoteForceInterface < TYPE, ROT, STORAGE, rfi_real_t >::Close() {
  if (! Active()) return;
  output("RFI: %s: Closing ...\n", name);
  if (TYPE == ForceIntegrator) {
   for (int i=0; i<workers; i++) {
    sizes[i] = RFI_FINISHED;
   }
  }
  SendSizes();
  if (TYPE == ForceIntegrator) {
   Finish();
  }
}

template < rfi_type_t TYPE, rfi_rot_t ROT, rfi_storage_t STORAGE, typename rfi_real_t >
void RemoteForceInterface < TYPE, ROT, STORAGE, rfi_real_t >::Finish() {
  if (! Active()) return;
  Zero();
  MPI_Barrier(intercomm);
  output("RFI: %s: Closed.\n", name);
  MPI_Comm_free(&intercomm);
  intercomm = MPI_COMM_NULL;
  connected = false;
  active = false;
}


template < rfi_type_t TYPE, rfi_rot_t ROT, rfi_storage_t STORAGE, typename rfi_real_t >
void RemoteForceInterface < TYPE, ROT, STORAGE, rfi_real_t >::SendSizes() {
  if (! Active()) return;
  debug1("RFI: %s: SendSizes ...\n", name);
  if (TYPE == ForceCalculator) {
    MPI_Alltoall(NULL, 0, MPI_SIZE_T, &sizes[0], 1, MPI_SIZE_T, intercomm);
    if (sizes[0] == RFI_FINISHED) {
     Finish();
    } else {
     Alloc();
    }
  } else {
    MPI_Alltoall(&sizes[0], 1, MPI_SIZE_T, NULL, 0, MPI_SIZE_T, intercomm);
  }
}


template < rfi_type_t TYPE, rfi_rot_t ROT, rfi_storage_t STORAGE, typename rfi_real_t >
void RemoteForceInterface < TYPE, ROT, STORAGE, rfi_real_t >::SendForces() {
    if (! Active()) return;
    debug1("RFI: %s: SendForces ...\n", name);
    for (int i=0; i<workers; i++) if (sizes[i] > 0) {
      if (TYPE == ForceCalculator) {
        MPI_Send(&Data(offsets[i],0), sizes[i], MPI_PARTICLE, i, 0xF1, intercomm);
      } else {
        MPI_Status stat;
        MPI_Recv(&Data(offsets[i],0), sizes[i], MPI_PARTICLE, i, 0xF1, intercomm, &stat);
      }
    }
}

template < rfi_type_t TYPE, rfi_rot_t ROT, rfi_storage_t STORAGE, typename rfi_real_t >
void RemoteForceInterface < TYPE, ROT, STORAGE, rfi_real_t >::SendParticles() {
    if (! Active()) return;
    debug1("RFI: %s: SendParticles ...\n", name);
    for (int i=0; i<workers; i++) if (sizes[i] > 0) {
      if (TYPE == ForceCalculator) {
        MPI_Status stat;
        MPI_Recv(&Data(offsets[i],0), sizes[i], MPI_PARTICLE, i, 0xF2, intercomm, &stat);
      } else {
        MPI_Send(&Data(offsets[i],0), sizes[i], MPI_PARTICLE, i, 0xF2, intercomm);
      }
    }
}

};

#ifdef RFI_DEF_ERROR
 #undef ERROR
#endif
#ifdef RFI_DEF_debug1
 #undef debug1
#endif
#ifdef RFI_DEF_output
 #undef output
#endif
