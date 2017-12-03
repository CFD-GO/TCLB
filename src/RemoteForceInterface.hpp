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

#define safe_MPI_Type_free(datatype) { if ((*datatype) != NULL) MPI_Type_free(datatype); }

template < rfi_type_t TYPE, rfi_rot_t ROT, rfi_storage_t STORAGE, typename rfi_real_t >
RemoteForceInterface < TYPE, ROT, STORAGE, rfi_real_t >::RemoteForceInterface() {
   workers = 0;
   masters = 0;
   intercomm = MPI_COMM_NULL;
   totsize = 0;
   ntab = 0;
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
void RemoteForceInterface < TYPE, ROT, STORAGE, rfi_real_t >::MakeTypes(bool particle_size_change, bool totsize_change) {
  bool commit_types = false;
  int parti_size = RFI_DATA_VOL - RFI_DATA_R;
  int force_size = RFI_DATA_SIZE - RFI_DATA_VOL;
  MPI_Datatype tmp;
  MPI_Aint lb=0;
  if (STORAGE == ArrayOfStructures) {
    if (particle_size_change) {
      safe_MPI_Type_free(&MPI_PARTICLE);
      safe_MPI_Type_free(&MPI_FORCES);
      MPI_Type_contiguous(parti_size, MPI_RFI_REAL_T, &tmp);
      MPI_Aint lb = 0;
      MPI_Type_create_resized(tmp, lb, real_size * RFI_DATA_SIZE, &MPI_PARTICLE);
      safe_MPI_Type_free(&tmp);
      MPI_Type_indexed(1, &force_size, &parti_size, MPI_RFI_REAL_T, &tmp); 
      MPI_Type_create_resized(tmp, real_size * parti_size, real_size * RFI_DATA_SIZE, &MPI_FORCES);
      safe_MPI_Type_free(&tmp);
      commit_types = true;
    }
  } else {
    if (totsize_change) {
      safe_MPI_Type_free(&MPI_PARTICLE);
      safe_MPI_Type_free(&MPI_FORCES);
      static int mt_offsets[RFI_DATA_SIZE], mt_sizes[RFI_DATA_SIZE];
      for (int i=0;i<RFI_DATA_SIZE;i++) { mt_offsets[i] = i*totsize; mt_sizes[i] = 1; }
      MPI_Type_indexed(parti_size, &mt_sizes[0], &mt_offsets[0], MPI_RFI_REAL_T, &tmp);
      MPI_Type_create_resized(tmp, lb, real_size, &MPI_PARTICLE);
      safe_MPI_Type_free(&tmp);
      MPI_Type_indexed(force_size, &mt_sizes[parti_size], &mt_offsets[parti_size], MPI_RFI_REAL_T, &tmp);
      MPI_Type_create_resized(tmp, lb, real_size, &MPI_FORCES);
      safe_MPI_Type_free(&tmp);
      commit_types = true;
    }
  }
  if (commit_types) {
   output("RFI: Adding type MPI ...\n");
   MPI_Type_commit(&MPI_PARTICLE);
   MPI_Type_commit(&MPI_FORCES);
//   #ifdef DEBUG_TYPES
    MPI_Aint lb,ex; int si;
    MPI_Type_get_extent(MPI_PARTICLE, &lb, &ex);
    MPI_Type_size(MPI_PARTICLE, &si);
    output("MPI_PARTICLE: size: %d, lb: %ld, ex: %ld\n", si, lb, ex);
    MPI_Type_get_extent(MPI_FORCES, &lb, &ex);
    MPI_Type_size(MPI_FORCES, &si);
    output("MPI_FORCES: size: %d, lb: %ld, ex: %ld\n", si, lb, ex);
//   #endif
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
   particle_size = RFI_DATA_SIZE;
   output("RFI: %s: Decided to calculate with rotation\n",name);
  } else {
   particle_size = RFI_DATA_SIZE;
   output("RFI: %s: Decided to calculate without rotation\n",name);
  }

   MPI_Aint lb;
   MPI_Type_get_extent(MPI_RFI_REAL_T, &lb, &real_size);
   MPI_PARTICLE = NULL;
   MPI_FORCES = NULL;

  MakeTypes(true,true);

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
      MakeTypes(false, true);      
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
        MPI_Send(&Data(offsets[i],0), sizes[i], MPI_FORCES, i, 0xF1, intercomm);
      } else {
        MPI_Status stat;
        MPI_Recv(&Data(offsets[i],0), sizes[i], MPI_FORCES, i, 0xF1, intercomm, &stat);
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
