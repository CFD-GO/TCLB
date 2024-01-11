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
#ifndef RFI_STRING_LEN
 #define RFI_STRING_LEN 1024*4
#endif

namespace rfi {

const int version = 0x000104;

#define safe_MPI_Type_free(datatype) { if ((*datatype) != NULL) MPI_Type_free(datatype); }

template <typename T> inline MPI_Datatype MPI_dt();
template <> inline MPI_Datatype MPI_dt< int >() { return MPI_INT; }
template <> inline MPI_Datatype MPI_dt< unsigned int >() { return MPI_UNSIGNED; }
template <> inline MPI_Datatype MPI_dt< long int >() { return MPI_LONG; }
template <> inline MPI_Datatype MPI_dt< unsigned long int >() { return MPI_UNSIGNED_LONG; }
template <> inline MPI_Datatype MPI_dt< char >() { return MPI_CHAR; }
template <> inline MPI_Datatype MPI_dt< float >() { return MPI_FLOAT; }
template <> inline MPI_Datatype MPI_dt< double >() { return MPI_DOUBLE; }


template < rfi_type_t TYPE, rfi_rot_t ROT, rfi_storage_t STORAGE, typename rfi_real_t, typename tab_allocator >
template <class T> inline T RemoteForceInterface< TYPE, ROT, STORAGE, rfi_real_t, tab_allocator >::Exchange(T out) {
   T in;
   MPI_Request request; MPI_Status status;
   MPI_Datatype datatype = MPI_dt<T>();
   if (rank == 0) {
      MPI_Isend(&out, 1, datatype, 0, 0xE0, intercomm, &request);
      MPI_Recv(&in, 1, datatype, 0, 0xE0, intercomm, &status);
      MPI_Wait(&request,  &status);
   }
   MPI_Bcast(&in, 1, datatype, 0, comm);
   return in;
};

template < rfi_type_t TYPE, rfi_rot_t ROT, rfi_storage_t STORAGE, typename rfi_real_t, typename tab_allocator >
template <class T> inline std::vector<T> RemoteForceInterface< TYPE, ROT, STORAGE, rfi_real_t, tab_allocator >::Exchange(std::vector<T> out) {
   std::vector<T> in;
   size_t in_size = Exchange(out.size());
   in.resize(in_size);
   MPI_Request request; MPI_Status status;
   MPI_Datatype datatype = MPI_dt<T>();
   if (rank == 0) {
      MPI_Isend(&out[0], out.size(), datatype, 0, 0xE1, intercomm, &request);
      MPI_Recv(&in[0], in.size(), datatype, 0, 0xE1, intercomm, &status);
      MPI_Wait(&request,  &status);
   }
   MPI_Bcast(&in[0], in.size(), datatype, 0, comm);
   return in;
};

template < rfi_type_t TYPE, rfi_rot_t ROT, rfi_storage_t STORAGE, typename rfi_real_t, typename tab_allocator >
template <class T> inline std::basic_string<T> RemoteForceInterface< TYPE, ROT, STORAGE, rfi_real_t, tab_allocator >::Exchange(std::basic_string<T> out) {
   std::basic_string<T> in;
   size_t in_size = Exchange(out.size());
   in.resize(in_size);
   MPI_Request request; MPI_Status status;
   MPI_Datatype datatype = MPI_dt<T>();
   if (rank == 0) {
      MPI_Isend(&out[0], out.size(), datatype, 0, 0xE1, intercomm, &request);
      MPI_Recv(&in[0], in.size(), datatype, 0, 0xE1, intercomm, &status);
      MPI_Wait(&request,  &status);
   }
   MPI_Bcast(&in[0], in.size(), datatype, 0, comm);
   return in;
};



template < rfi_type_t TYPE, rfi_rot_t ROT, rfi_storage_t STORAGE, typename rfi_real_t, typename tab_allocator >
RemoteForceInterface < TYPE, ROT, STORAGE, rfi_real_t, tab_allocator >::RemoteForceInterface() {
   workers = 0;
   masters = 0;
   intercomm = MPI_COMM_NULL;
   totsize = 0;
   ntab = 0;
   connected = false;
   active = false;
   my_type = TYPE;
   stats = false;
   stats_iter = 0;
   kill_flag = 666;
   alreadyKilledEverybody = false;
   myBox.declared = false;
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

template < rfi_type_t TYPE, rfi_rot_t ROT, rfi_storage_t STORAGE, typename rfi_real_t, typename tab_allocator >
RemoteForceInterface < TYPE, ROT, STORAGE, rfi_real_t, tab_allocator >::~RemoteForceInterface() {
  debug1("RFI: Req left in the buffers: sizes: %d particles: %d forces: %d\n", sizes_req.size(), particles_req.size(), forces_req.size());
  if (intercomm != MPI_COMM_NULL) {
    ERROR("RFI: This should never happen\n"); // LCOV_EXCL_LINE
  }
}

template < rfi_type_t TYPE, rfi_rot_t ROT, rfi_storage_t STORAGE, typename rfi_real_t, typename tab_allocator >
void RemoteForceInterface < TYPE, ROT, STORAGE, rfi_real_t, tab_allocator >::enableStats(const char * filename, int iter) {
  stats = true;
  if (filename == NULL) {
    stats_prefix = "";
  } else {
    stats_prefix = filename;
  }
  if (iter < 1) iter = 1;
  stats_iter = iter;
  if (connected) allocStats();
}

template < rfi_type_t TYPE, rfi_rot_t ROT, rfi_storage_t STORAGE, typename rfi_real_t, typename tab_allocator >
void RemoteForceInterface < TYPE, ROT, STORAGE, rfi_real_t, tab_allocator >::allocStats() {
   sizesStats.resize(workers, 0);
   sizesStatsNum = 0;
   waitStats.resize(12, 0);
   waitStatsNum.resize(12, 0);
   char fn[RFI_STRING_LEN];
   
   if (stats_prefix == "") stats_prefix = "RFI";
   sprintf(fn, "%s_%s_P%02d.txt", stats_prefix.c_str(), name.c_str(), rank);
   stats_filename = fn;

   FILE * f = fopen(stats_filename.c_str(), "w");
   fprintf(f,"size_iter");
   for (int i=0; i<workers; i++) fprintf(f,", size_%03d", i);
   for (int i=0; i<12; i++) fprintf(f,", dt_%02d", i);
   fprintf(f,"\n");
   fclose(f);
}

template < rfi_type_t TYPE, rfi_rot_t ROT, rfi_storage_t STORAGE, typename rfi_real_t, typename tab_allocator >
void RemoteForceInterface < TYPE, ROT, STORAGE, rfi_real_t, tab_allocator >::saveSizesStats() {
   for (int i=0; i<workers; i++) sizesStats[i] += sizes[i];
   sizesStatsNum += 1;
}

template < rfi_type_t TYPE, rfi_rot_t ROT, rfi_storage_t STORAGE, typename rfi_real_t, typename tab_allocator >
void RemoteForceInterface < TYPE, ROT, STORAGE, rfi_real_t, tab_allocator >::saveWaitStats(int index) {
  double t = MPI_Wtime();
  static double t0=-0.123;
  static int index0 = -1;
  if (t0 != -0.123) {
    double dt = t - t0;
    waitStats[index] += dt;
    waitStatsNum[index] += 1;
  }
  if (index == 0) t0 = t;
}

template < rfi_type_t TYPE, rfi_rot_t ROT, rfi_storage_t STORAGE, typename rfi_real_t, typename tab_allocator >
void RemoteForceInterface < TYPE, ROT, STORAGE, rfi_real_t, tab_allocator >::printStats() {
  if (sizesStatsNum == stats_iter) {
   char fn[RFI_STRING_LEN];
   sprintf(fn, "RFI_stats_%s_%d.txt", name.c_str(), rank);
   FILE * f = fopen(stats_filename.c_str(), "a");
   fprintf(f,"%ld", sizesStatsNum);
   for (int i=0; i<workers; i++) fprintf(f,", %lg", (double) sizesStats[i] / sizesStatsNum);
   for (int i=0; i<12; i++) fprintf(f,", %lg", (double) waitStats[i] / waitStatsNum[i]);
   fprintf(f,"\n");
   fclose(f);
   for (int i=0; i<workers; i++) sizesStats[i] = 0.0;
   sizesStatsNum = 0;
   for (int i=0; i<12; i++) { waitStats[i] = 0.0; waitStatsNum[i] = 0.0; }
  }
}


template < rfi_type_t TYPE, rfi_rot_t ROT, rfi_storage_t STORAGE, typename rfi_real_t, typename tab_allocator >
inline void RemoteForceInterface < TYPE, ROT, STORAGE, rfi_real_t, tab_allocator >::setUnits(rfi_real_t meter, rfi_real_t second, rfi_real_t kilogram) {
 if (Connected()) {
   ERROR("Units can be set only before connection is established\n");
   exit(-1);
 }
 base_units[0] = meter;
 base_units[1] = second;
 base_units[2] = kilogram;
 non_trivial_units = true;
}

template < rfi_type_t TYPE, rfi_rot_t ROT, rfi_storage_t STORAGE, typename rfi_real_t, typename tab_allocator >
inline void RemoteForceInterface < TYPE, ROT, STORAGE, rfi_real_t, tab_allocator >::CanCopeWithUnits(bool ccwu_) {
   if (Connected()) {
     ERROR("You can set the can_cope_with_units flag only before connection is established\n");
     exit(-1);
   }
   can_cope_with_units = ccwu_;
}

template < rfi_type_t TYPE, rfi_rot_t ROT, rfi_storage_t STORAGE, typename rfi_real_t, typename tab_allocator >
void RemoteForceInterface < TYPE, ROT, STORAGE, rfi_real_t, tab_allocator >::MakeTypes(bool particle_size_change, bool totsize_change) {
  bool commit_types = false;
  int parti_size = RFI_DATA_IN - RFI_DATA_START;
  int force_size = RFI_DATA_SIZE - RFI_DATA_IN;
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


template < rfi_type_t TYPE, rfi_rot_t ROT, rfi_storage_t STORAGE, typename rfi_real_t, typename tab_allocator >
int RemoteForceInterface < TYPE, ROT, STORAGE, rfi_real_t, tab_allocator >::Negotiate() {
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
//      unit[RFI_DATA_VOL] = meter*meter*meter;
      for (int i=0;i<3;i++) {
        unit[RFI_DATA_POS+i] = meter;
        unit[RFI_DATA_VEL+i] = meter/second;
        unit[RFI_DATA_ANGVEL+i] = 1.0/second;
        unit[RFI_DATA_FORCE+i] = kilogram*meter/(second*second);
        unit[RFI_DATA_MOMENT+i] = kilogram*meter*meter/(second*second);
      }
    }
  }

  int my_stats = stats;
  int other_stats = Exchange(my_stats);
  stats = my_stats || other_stats;
  
  std::string my_stats_prefix = stats_prefix;
  std::string other_stats_prefix = Exchange(my_stats_prefix);
  
  if (my_stats_prefix == "") {
    stats_prefix = other_stats_prefix;
  }

  int my_stats_iter = stats_iter;
  int other_stats_iter = Exchange(my_stats_iter);
  
  if (my_stats_iter == 0) {
    stats_iter = other_stats_iter;
  }

  if (stats) {
    allocStats();
    output("RFI: %s: Decided to calculate with statistics\n",name.c_str());
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


template < rfi_type_t TYPE, rfi_rot_t ROT, rfi_storage_t STORAGE, typename rfi_real_t, typename tab_allocator >
int RemoteForceInterface < TYPE, ROT, STORAGE, rfi_real_t, tab_allocator >::Connect(MPI_Comm comm_, MPI_Comm intercomm_) {
   int ret = 0;
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
   if (stats) allocStats();
   ret = Negotiate();
   if (ret) return ret;
   
   // Prepare stop mechanics
   size_t death_size = 1;
   size_t death_i = 0;
   if (rank == 0) death_size += 1 + masters;
   MPI_Request req;
   death_flag.resize(death_size);
   MPI_Irecv(&death_flag[death_i], 1, MPI_INT, 0, 0xD0, comm, &req);
   death_req.push_back(req);
   death_i++;
   if (rank == 0) {
     MPI_Irecv(&death_flag[death_i], 1, MPI_INT, 0, 0xD1, intercomm, &req);
     death_req.push_back(req);
     death_i++;
     for (int i=0; i<masters; i++) {
       MPI_Irecv(&death_flag[death_i], 1, MPI_INT, i, 0xD2, comm, &req);
       death_req.push_back(req);
       death_i++;
     }
   }
   ExchangeBoxes();
   return 0;
}

template < rfi_type_t TYPE, rfi_rot_t ROT, rfi_storage_t STORAGE, typename rfi_real_t, typename tab_allocator >
void RemoteForceInterface < TYPE, ROT, STORAGE, rfi_real_t, tab_allocator >::ExchangeBoxes() {
  std::vector< MPI_Request > reqs;
  MPI_Request req;
  workerBoxes.resize(workers);
  for (int i = 0; i < workers; i++) {
    MPI_Isend(&myBox,          sizeof(Box), MPI_BYTE, i, 0xC0, intercomm, &req);
    MPI_Irecv(&workerBoxes[i], sizeof(Box), MPI_BYTE, i, 0xC0, intercomm, &req);
    reqs.push_back(req);
  }
  WaitAll(reqs);
}


template < rfi_type_t TYPE, rfi_rot_t ROT, rfi_storage_t STORAGE, typename rfi_real_t, typename tab_allocator >
void RemoteForceInterface < TYPE, ROT, STORAGE, rfi_real_t, tab_allocator >::Alloc() {
    offsets[0] = 0;
    for (int i=0; i<workers; i++) offsets[i+1] = offsets[i] + sizes[i];
    if (totsize != offsets[workers]) {
      totsize = offsets[workers];
      ntab = totsize * particle_size;
      if (ntab > tab.size()) tab.resize(ntab);
      MakeTypes(false, true);      
    }
}


template < rfi_type_t TYPE, rfi_rot_t ROT, rfi_storage_t STORAGE, typename rfi_real_t, typename tab_allocator >
void RemoteForceInterface < TYPE, ROT, STORAGE, rfi_real_t, tab_allocator >::Zero() {
  totsize = 0;
  for (int i=0; i<workers; i++) {
   sizes[i] = 0;
   offsets[i] = 0;
  }
}

template < rfi_type_t TYPE, rfi_rot_t ROT, rfi_storage_t STORAGE, typename rfi_real_t, typename tab_allocator >
void RemoteForceInterface < TYPE, ROT, STORAGE, rfi_real_t, tab_allocator >::Close() {
  if (! Active()) return;
  debug1("RFI: %s: Sending the order to kill ...\n", name.c_str());
  MPI_Request req;
  MPI_Isend(&kill_flag, 1, MPI_INT, 0, 0xD2, comm, &req); // kill root
  if (rank == 0) KillEverybody();
  debug1("RFI: %s: Waiting for death ...\n", name.c_str());
  WaitForDeath();
}

template < rfi_type_t TYPE, rfi_rot_t ROT, rfi_storage_t STORAGE, typename rfi_real_t, typename tab_allocator >
void RemoteForceInterface < TYPE, ROT, STORAGE, rfi_real_t, tab_allocator >::WaitAll(std::vector<MPI_Request>& reqs) {
  static std::vector<MPI_Status> status_vec;
  int ind = -1;
  if (reqs.size() < 1) return;
  int reqs_len = reqs.size();
  for (size_t i=0; i<death_req.size(); i++) reqs.push_back(death_req[i]);
  status_vec.resize(reqs.size());
  for (int i = 0; i < reqs_len; i++) {
    MPI_Waitany(reqs.size(), &reqs[0], &ind, &status_vec[0]);
    if (ind == reqs_len) { Death(); break; }
    if (ind > reqs_len) { KillEverybody(); WaitForDeath(); return;}
  }
  reqs.clear();
}


template < rfi_type_t TYPE, rfi_rot_t ROT, rfi_storage_t STORAGE, typename rfi_real_t, typename tab_allocator >
void RemoteForceInterface < TYPE, ROT, STORAGE, rfi_real_t, tab_allocator >::Death() {
  debug1("RFI: %s: Death ...\n", name.c_str());
  if (! Active()) return;
  Zero();
  MPI_Barrier(intercomm);
  output("RFI: %s: Closed.\n", name.c_str());
  MPI_Comm_free(&intercomm);
  intercomm = MPI_COMM_NULL;
  connected = false;
  active = false;
}

template < rfi_type_t TYPE, rfi_rot_t ROT, rfi_storage_t STORAGE, typename rfi_real_t, typename tab_allocator >
void RemoteForceInterface < TYPE, ROT, STORAGE, rfi_real_t, tab_allocator >::WaitForDeath() {
/*  std::vector<MPI_Status> status_vec;
  int ind=-1;
  status_vec.resize(death_req.size());
  while (true) {
    MPI_Waitany(death_req.size(), &death_req[0], &ind, &status_vec[0]);
    if (ind == 0) { Death(); break; }
    if (ind > 0) { KillEverybody(); }
  }
  return; */
  MPI_Status status;
  MPI_Wait(&death_req[0], &status);
  Death();
}

template < rfi_type_t TYPE, rfi_rot_t ROT, rfi_storage_t STORAGE, typename rfi_real_t, typename tab_allocator >
void RemoteForceInterface < TYPE, ROT, STORAGE, rfi_real_t, tab_allocator >::KillEverybody() {
  if (alreadyKilledEverybody) return;
  debug1("RFI: %s: Killing everygody ...\n", name.c_str());
  MPI_Request req;
  MPI_Isend(&kill_flag, 1, MPI_INT, 0, 0xD1, intercomm, &req); // kill partner
  for (int i = 0; i < masters; i++) {
    MPI_Isend(&kill_flag, 1, MPI_INT, i, 0xD0, comm, &req); // kill siblings
  }
  alreadyKilledEverybody = true;
}


template < rfi_type_t TYPE, rfi_rot_t ROT, rfi_storage_t STORAGE, typename rfi_real_t, typename tab_allocator >
void RemoteForceInterface < TYPE, ROT, STORAGE, rfi_real_t, tab_allocator >::ISendSizes() {
  debug1("RFI: %s:   ISendSizes ...\n", name.c_str());
  if (stats) printStats();
  if (stats) saveWaitStats(0);
  for (int i=0; i<workers; i++) {
   MPI_Request req;
   if (TYPE == ForceCalculator) {
     MPI_Irecv(&sizes[i], 1, MPI_SIZE_T, i, 0xF0, intercomm, &req);
   } else {
     MPI_Isend(&sizes[i], 1, MPI_SIZE_T, i, 0xF0, intercomm, &req);
   }
   sizes_req.push_back(req);
  }
  if (stats) saveWaitStats(1);
}

template < rfi_type_t TYPE, rfi_rot_t ROT, rfi_storage_t STORAGE, typename rfi_real_t, typename tab_allocator >
void RemoteForceInterface < TYPE, ROT, STORAGE, rfi_real_t, tab_allocator >::WSendSizes() {
  debug1("RFI: %s:   WSendSizes ...\n", name.c_str());
  if (stats) saveWaitStats(2);
  WaitAll(sizes_req);
  if (stats) saveWaitStats(3);
}


template < rfi_type_t TYPE, rfi_rot_t ROT, rfi_storage_t STORAGE, typename rfi_real_t, typename tab_allocator >
void RemoteForceInterface < TYPE, ROT, STORAGE, rfi_real_t, tab_allocator >::SendSizes() {
  if (! Active()) return;
  debug1("RFI: %s: SendSizes {\n", name.c_str());
  if (TYPE == ForceCalculator) {
    WSendForces();
    if (! Active()) return;
    WSendSizes();
    if (! Active()) return;
    Alloc();
    ISendParticles();
  } else {
    ISendSizes();
  }
  debug1("RFI: %s: } // SendSizes\n", name.c_str());
}


template < rfi_type_t TYPE, rfi_rot_t ROT, rfi_storage_t STORAGE, typename rfi_real_t, typename tab_allocator >
void RemoteForceInterface < TYPE, ROT, STORAGE, rfi_real_t, tab_allocator >::ISendForces() {
    debug1("RFI: %s:   ISendForces ...\n", name.c_str());
    if (stats) saveWaitStats(8);
    for (int i=0; i<workers; i++) if (sizes[i] > 0) {
      MPI_Request req;
      if (TYPE == ForceCalculator) {
        MPI_Isend(&RawData(offsets[i],0), sizes[i], MPI_FORCES, i, 0xF1, intercomm, &req);
      } else {
        MPI_Irecv(&RawData(offsets[i],0), sizes[i], MPI_FORCES, i, 0xF1, intercomm, &req);
      }
      forces_req.push_back(req);
    }
    if (stats) saveWaitStats(9);
}

template < rfi_type_t TYPE, rfi_rot_t ROT, rfi_storage_t STORAGE, typename rfi_real_t, typename tab_allocator >
void RemoteForceInterface < TYPE, ROT, STORAGE, rfi_real_t, tab_allocator >::WSendForces() {
  debug1("RFI: %s:   WSendForces ...\n", name.c_str());
    if (stats) saveWaitStats(10);
  WaitAll(forces_req);
    if (stats) saveWaitStats(11);
}


template < rfi_type_t TYPE, rfi_rot_t ROT, rfi_storage_t STORAGE, typename rfi_real_t, typename tab_allocator >
void RemoteForceInterface < TYPE, ROT, STORAGE, rfi_real_t, tab_allocator >::SendForces() {
    if (! Active()) return;
    debug1("RFI: %s: SendForces {\n", name.c_str());
    if (TYPE == ForceCalculator) {
        ISendForces();
        ISendSizes();
    } else {
        WSendSizes();
        if (! Active()) return;
        WSendParticles();
        if (! Active()) return;
        WSendForces();
        if (! Active()) return;
    }
    debug1("RFI: %s: } // SendForces\n", name.c_str());
}

template < rfi_type_t TYPE, rfi_rot_t ROT, rfi_storage_t STORAGE, typename rfi_real_t, typename tab_allocator >
void RemoteForceInterface < TYPE, ROT, STORAGE, rfi_real_t, tab_allocator >::ISendParticles() {
    debug1("RFI: %s:   ISendParticles ...\n", name.c_str());
    if (stats) saveSizesStats();
    if (stats) saveWaitStats(4);
    for (int i=0; i<workers; i++) if (sizes[i] > 0) {
      MPI_Request req;
      if (TYPE == ForceCalculator) {
        MPI_Irecv(&RawData(offsets[i],0), sizes[i], MPI_PARTICLE, i, 0xF2, intercomm, &req);
      } else {
        MPI_Isend(&RawData(offsets[i],0), sizes[i], MPI_PARTICLE, i, 0xF2, intercomm, &req);
      }
      particles_req.push_back(req);
    }
    if (stats) saveWaitStats(5);
}

template < rfi_type_t TYPE, rfi_rot_t ROT, rfi_storage_t STORAGE, typename rfi_real_t, typename tab_allocator >
void RemoteForceInterface < TYPE, ROT, STORAGE, rfi_real_t, tab_allocator >::WSendParticles() {
    debug1("RFI: %s:   WSendParticles ...\n", name.c_str());
    if (stats) saveWaitStats(6);
    WaitAll(particles_req);
    if (stats) saveWaitStats(7);
}

template < rfi_type_t TYPE, rfi_rot_t ROT, rfi_storage_t STORAGE, typename rfi_real_t, typename tab_allocator >
void RemoteForceInterface < TYPE, ROT, STORAGE, rfi_real_t, tab_allocator >::SendParticles() {
    if (! Active()) return;
    debug1("RFI: %s: SendParticles {\n", name.c_str());
    if (TYPE == ForceCalculator) {
      WSendParticles();
      if (! Active()) return;
    } else {
      ISendParticles();
      ISendForces();
    }
    debug1("RFI: %s: } // SendParticles\n", name.c_str());
}

template < rfi_type_t TYPE, rfi_rot_t ROT, rfi_storage_t STORAGE, typename rfi_real_t, typename tab_allocator >
void RemoteForceInterface < TYPE, ROT, STORAGE, rfi_real_t, tab_allocator >::DeclareSimpleBox(rfi_real_t x0, rfi_real_t x1, rfi_real_t y0, rfi_real_t y1, rfi_real_t z0, rfi_real_t z1) {
  myBox.declared = true;
  myBox.lower[0] = x0 / base_units[0];
  myBox.lower[1] = y0 / base_units[0];
  myBox.lower[2] = z0 / base_units[0];
  myBox.upper[0] = x1 / base_units[0];
  myBox.upper[1] = y1 / base_units[0];
  myBox.upper[2] = z1 / base_units[0];
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
