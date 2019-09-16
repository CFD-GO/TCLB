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

const int version = 0x000102;

#define safe_MPI_Type_free(datatype) { if ((*datatype) != NULL) MPI_Type_free(datatype); }

template <typename T> inline MPI_Datatype MPI_dt();
template <> inline MPI_Datatype MPI_dt< int >() { return MPI_INT; }
template <> inline MPI_Datatype MPI_dt< unsigned int >() { return MPI_UNSIGNED; }
template <> inline MPI_Datatype MPI_dt< long int >() { return MPI_LONG; }
template <> inline MPI_Datatype MPI_dt< unsigned long int >() { return MPI_UNSIGNED_LONG; }
template <> inline MPI_Datatype MPI_dt< char >() { return MPI_CHAR; }
template <> inline MPI_Datatype MPI_dt< float >() { return MPI_FLOAT; }
template <> inline MPI_Datatype MPI_dt< double >() { return MPI_DOUBLE; }


template < rfi_type_t TYPE, rfi_rot_t ROT, rfi_storage_t STORAGE, typename rfi_real_t >
template <class T> inline T RemoteForceInterface< TYPE, ROT, STORAGE, rfi_real_t >::Exchange(T out) {
   T in;
   MPI_Request request; MPI_Status status;
   MPI_Datatype datatype = MPI_dt<T>();
   if (rank == 0) {
      MPI_Isend(&out, 1, datatype, 0, 123, intercomm, &request);
      MPI_Recv(&in, 1, datatype, 0, 123, intercomm, &status);
      MPI_Wait(&request,  &status);
   }
   MPI_Bcast(&in, 1, datatype, 0, comm);
   return in;
};

template < rfi_type_t TYPE, rfi_rot_t ROT, rfi_storage_t STORAGE, typename rfi_real_t >
template <class T> inline std::vector<T> RemoteForceInterface< TYPE, ROT, STORAGE, rfi_real_t >::Exchange(std::vector<T> out) {
   std::vector<T> in;
   size_t in_size = Exchange(out.size());
   in.resize(in_size);
   MPI_Request request; MPI_Status status;
   MPI_Datatype datatype = MPI_dt<T>();
   if (rank == 0) {
      MPI_Isend(&out[0], out.size(), datatype, 0, 124, intercomm, &request);
      MPI_Recv(&in[0], in.size(), datatype, 0, 124, intercomm, &status);
      MPI_Wait(&request,  &status);
   }
   MPI_Bcast(&in[0], in.size(), datatype, 0, comm);
   return in;
};



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
     ERROR("RFI: Unknown type of RemoteForceInterface");
     exit(-1);
   }
   particle_size = 0;
   MPI_RFI_REAL_T = MPI_dt< rfi_real_t >();
   switch (sizeof(rfi_real_t)) {
   case 8:
   case 4:
     break;
   default:   
     ERROR("RFI: Unknown type rfi_real_t in RemoteForceInterface");
     exit(-1);
   }
   base_units[0] = 1.0;
   base_units[1] = 1.0;
   base_units[2] = 1.0;
   non_trivial_units = false;
   can_cope_with_units = true;
}

template < rfi_type_t TYPE, rfi_rot_t ROT, rfi_storage_t STORAGE, typename rfi_real_t >
RemoteForceInterface < TYPE, ROT, STORAGE, rfi_real_t >::~RemoteForceInterface() {
  debug1("RFI: Req left in the buffers: sizes: %d particles: %d forces: %d\n", sizes_req.size(), particles_req.size(), forces_req.size());
  if (intercomm != MPI_COMM_NULL) {
    ERROR("RFI: This should never happen\n"); // LCOV_EXCL_LINE
  }
}

template < rfi_type_t TYPE, rfi_rot_t ROT, rfi_storage_t STORAGE, typename rfi_real_t >
inline void RemoteForceInterface < TYPE, ROT, STORAGE, rfi_real_t >::setUnits(rfi_real_t meter, rfi_real_t second, rfi_real_t kilogram) {
 if (Connected()) {
   ERROR("Units can be set only before connection is established\n");
   exit(-1);
 }
 base_units[0] = meter;
 base_units[1] = second;
 base_units[2] = kilogram;
 non_trivial_units = true;
}

template < rfi_type_t TYPE, rfi_rot_t ROT, rfi_storage_t STORAGE, typename rfi_real_t >
inline void RemoteForceInterface < TYPE, ROT, STORAGE, rfi_real_t >::CanCopeWithUnits(bool ccwu_) {
   if (Connected()) {
     ERROR("You can set the can_cope_with_units flag only before connection is established\n");
     exit(-1);
   }
   can_cope_with_units = ccwu_;
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
  }
}


template < rfi_type_t TYPE, rfi_rot_t ROT, rfi_storage_t STORAGE, typename rfi_real_t >
int RemoteForceInterface < TYPE, ROT, STORAGE, rfi_real_t >::Negotiate() {
  if (! connected) return -1;
  MPI_Barrier(intercomm);
  output("RFI: %s: Starting negotiations ...\n", name.c_str());
  MPI_Barrier(intercomm);
  
  int other_version = Exchange(version);
  if (version != other_version) {
    ERROR("RFI: RemoteForceInterface version mismatch. Exiting.\n");
    exit(-1);
  }

  int my_rot = ROT == RotParticle;
  int other_rot = Exchange(my_rot);
  rot = my_rot && other_rot;

  if (rot) {
   particle_size = RFI_DATA_SIZE;
   output("RFI: %s: Decided to calculate with rotation\n",name.c_str());
  } else {
   particle_size = RFI_DATA_SIZE;
   output("RFI: %s: Decided to calculate without rotation\n",name.c_str());
  }

  unit.resize(particle_size);
  for (int i=0;i<RFI_DATA_SIZE; i++) unit[i] = 1.0;


  MPI_Aint lb,ex; int si, other_si;
  MPI_Type_size(MPI_RFI_REAL_T, &si);
  real_size = si;
  other_si = Exchange(si);
  if (si != other_si) {
    ERROR("RFI: Sizes of float type mismatch\n");
    exit(-1);
  }
  MPI_PARTICLE = NULL;
  MPI_FORCES = NULL;

  MakeTypes(true,true);
  
  MPI_Type_get_extent(MPI_PARTICLE, &lb, &ex);
  MPI_Type_size(MPI_PARTICLE, &si);
  output("RFI: MPI_PARTICLE: size: %d, lb: %ld, ex: %ld\n", si, lb, ex);
  other_si = Exchange(si);
  if (si != other_si) {
    ERROR("RFI: Sizes of particle data mismatch\n");
    exit(-1);
  }
  MPI_Type_get_extent(MPI_FORCES, &lb, &ex);
  MPI_Type_size(MPI_FORCES, &si);
  output("RFI: MPI_FORCES: size: %d, lb: %ld, ex: %ld\n", si, lb, ex);
  other_si = Exchange(si);
  if (si != other_si) {
    ERROR("RFI: Sizes of force data mismatch\n");
    exit(-1);
  }
  
  int my_ntu = non_trivial_units;
  int other_ntu = Exchange(my_ntu);
  if (non_trivial_units || other_ntu) {
    output("RFI: %s: Non trivial units\n",name.c_str());
    int my_ccwu = can_cope_with_units;
    int other_ccwu = Exchange(my_ccwu);
    if (my_ccwu && other_ccwu) {
      if (TYPE == ForceCalculator) {
        other_ccwu = false;
      } else {
        my_ccwu = false;
      }
    }
    if (my_ccwu) {
      output("RFI: %s: I'm taking care of the units\n",name.c_str());
    } else if (other_ccwu) {
      // The other side is taking care of the units
    } else {
      ERROR("RFI: Nobody is taking care of the units!\n");
      exit(-1);
    }
    std::vector< rfi_real_t > my_units, other_units;
    for (int i=0;i<3;i++) my_units.push_back(base_units[i]);
    other_units = Exchange(my_units);
    double meter, second, kilogram;
    unit.resize(particle_size);
    if (my_ccwu) {
      meter = other_units[0]/my_units[0];
      second = other_units[1]/my_units[1];
      kilogram = other_units[2]/my_units[2];
      output("RFI: %s: Unit conversion: m:%lg, s:%lg, kg:%lg\n",name.c_str(), meter, second, kilogram);
      unit[RFI_DATA_R] = meter;
      unit[RFI_DATA_VOL] = meter*meter*meter;
      for (int i=0;i<3;i++) {
        unit[RFI_DATA_POS+i] = meter;
        unit[RFI_DATA_VEL+i] = meter/second;
        unit[RFI_DATA_ANGVEL+i] = 1.0/second;
        unit[RFI_DATA_FORCE+i] = kilogram*meter/(second*second);
        unit[RFI_DATA_MOMENT+i] = kilogram*meter*meter/(second*second);
      }
    }
  }

  MPI_Barrier(intercomm);
  output("RFI: %s: Finished negotiations\n",name.c_str());
  MPI_Barrier(intercomm);
  active = true;

    if (TYPE == ForceCalculator) {
        ISendSizes();
    }

  return 0;
}


template < rfi_type_t TYPE, rfi_rot_t ROT, rfi_storage_t STORAGE, typename rfi_real_t >
int RemoteForceInterface < TYPE, ROT, STORAGE, rfi_real_t >::Connect(MPI_Comm comm_, MPI_Comm intercomm_) {
   if (connected) {
     ERROR("RFI: Already connected");
     return -1;
   }
   comm = comm_;
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
  output("RFI: %s: Closing ...\n", name.c_str());
  if (TYPE == ForceIntegrator) {
   for (int i=0; i<workers; i++) {
    sizes[i] = RFI_FINISHED;
   }
  }
  ISendSizes();
  WSendSizes();
  if (TYPE == ForceIntegrator) {
   Finish();
  }
}

template < rfi_type_t TYPE, rfi_rot_t ROT, rfi_storage_t STORAGE, typename rfi_real_t >
void RemoteForceInterface < TYPE, ROT, STORAGE, rfi_real_t >::Finish() {
  if (! Active()) return;
  Zero();
  MPI_Barrier(intercomm);
  output("RFI: %s: Closed.\n", name.c_str());
  MPI_Comm_free(&intercomm);
  intercomm = MPI_COMM_NULL;
  connected = false;
  active = false;
}


template < rfi_type_t TYPE, rfi_rot_t ROT, rfi_storage_t STORAGE, typename rfi_real_t >
void RemoteForceInterface < TYPE, ROT, STORAGE, rfi_real_t >::WaitAll(std::vector<MPI_Request>& reqs) {
  static std::vector<MPI_Status> stats;
  if (reqs.size() < 1) return;
  stats.resize(reqs.size());
  MPI_Waitall(reqs.size(), &reqs[0], &stats[0]);
  reqs.clear();
}


template < rfi_type_t TYPE, rfi_rot_t ROT, rfi_storage_t STORAGE, typename rfi_real_t >
void RemoteForceInterface < TYPE, ROT, STORAGE, rfi_real_t >::ISendSizes() {
  debug1("RFI: %s: ISendSizes ...\n", name.c_str());
  for (int i=0; i<workers; i++) {
   MPI_Request req;
   if (TYPE == ForceCalculator) {
     MPI_Irecv(&sizes[i], 1, MPI_SIZE_T, i, 0xF0, intercomm, &req);
   } else {
     MPI_Isend(&sizes[i], 1, MPI_SIZE_T, i, 0xF0, intercomm, &req);
   }
   sizes_req.push_back(req);
  }
}

template < rfi_type_t TYPE, rfi_rot_t ROT, rfi_storage_t STORAGE, typename rfi_real_t >
void RemoteForceInterface < TYPE, ROT, STORAGE, rfi_real_t >::WSendSizes() {
  debug1("RFI: %s: WSendSizes ...\n", name.c_str());
  WaitAll(sizes_req);
}


template < rfi_type_t TYPE, rfi_rot_t ROT, rfi_storage_t STORAGE, typename rfi_real_t >
void RemoteForceInterface < TYPE, ROT, STORAGE, rfi_real_t >::SendSizes() {
  if (! Active()) return;
  debug1("RFI: %s: SendSizes ...\n", name.c_str());
  if (TYPE == ForceCalculator) {
    WSendForces();
    WSendSizes();
    if (sizes[0] == RFI_FINISHED) {
     Finish();
    } else {
     Alloc();
     ISendParticles();
    }
  } else {
    ISendSizes();
  }
}


template < rfi_type_t TYPE, rfi_rot_t ROT, rfi_storage_t STORAGE, typename rfi_real_t >
void RemoteForceInterface < TYPE, ROT, STORAGE, rfi_real_t >::ISendForces() {
    debug1("RFI: %s: ISendForces ...\n", name.c_str());
    for (int i=0; i<workers; i++) if (sizes[i] > 0) {
      MPI_Request req;
      if (TYPE == ForceCalculator) {
        MPI_Isend(&RawData(offsets[i],0), sizes[i], MPI_FORCES, i, 0xF1, intercomm, &req);
      } else {
        MPI_Irecv(&RawData(offsets[i],0), sizes[i], MPI_FORCES, i, 0xF1, intercomm, &req);
      }
      forces_req.push_back(req);
    }
}

template < rfi_type_t TYPE, rfi_rot_t ROT, rfi_storage_t STORAGE, typename rfi_real_t >
void RemoteForceInterface < TYPE, ROT, STORAGE, rfi_real_t >::WSendForces() {
  debug1("RFI: %s: WSendForces ...\n", name.c_str());
  WaitAll(forces_req);
}


template < rfi_type_t TYPE, rfi_rot_t ROT, rfi_storage_t STORAGE, typename rfi_real_t >
void RemoteForceInterface < TYPE, ROT, STORAGE, rfi_real_t >::SendForces() {
    if (! Active()) return;
    debug1("RFI: %s: SendForces ...\n", name.c_str());
    if (TYPE == ForceCalculator) {
        ISendForces();
        ISendSizes();
    } else {
        WSendSizes();
        WSendParticles();
        WSendForces();
    }
}

template < rfi_type_t TYPE, rfi_rot_t ROT, rfi_storage_t STORAGE, typename rfi_real_t >
void RemoteForceInterface < TYPE, ROT, STORAGE, rfi_real_t >::ISendParticles() {
    debug1("RFI: %s: ISendParticles ...\n", name.c_str());
    for (int i=0; i<workers; i++) if (sizes[i] > 0) {
      MPI_Request req;
      if (TYPE == ForceCalculator) {
        MPI_Irecv(&RawData(offsets[i],0), sizes[i], MPI_PARTICLE, i, 0xF2, intercomm, &req);
      } else {
        MPI_Isend(&RawData(offsets[i],0), sizes[i], MPI_PARTICLE, i, 0xF2, intercomm, &req);
      }
      particles_req.push_back(req);
    }
}

template < rfi_type_t TYPE, rfi_rot_t ROT, rfi_storage_t STORAGE, typename rfi_real_t >
void RemoteForceInterface < TYPE, ROT, STORAGE, rfi_real_t >::WSendParticles() {
    debug1("RFI: %s: WSendParticles ...\n", name.c_str());
    WaitAll(particles_req);
}

template < rfi_type_t TYPE, rfi_rot_t ROT, rfi_storage_t STORAGE, typename rfi_real_t >
void RemoteForceInterface < TYPE, ROT, STORAGE, rfi_real_t >::SendParticles() {
    if (! Active()) return;
    debug1("RFI: %s: SendParticles ...\n", name.c_str());
    if (TYPE == ForceCalculator) {
      WSendParticles();
    } else {
      ISendParticles();
      ISendForces();
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
