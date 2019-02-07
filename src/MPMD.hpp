#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <string>
#include <mpi.h>
#include <assert.h>
#include <map> //JM
#include <string.h> //JM

struct MPMDIntercomm {
   MPI_Comm local;
   MPI_Comm work;
   int local_size;
   int work_size;
   bool in_world;
   bool in_local;
   bool in_work;
   std::string name;
   bool connected;
   inline MPMDIntercomm() {
      local = work = MPI_COMM_NULL;
      local_size = work_size = 0;
      connected = in_world = in_local = in_work = false;
      name = "";
   }
   inline operator bool() const { return connected; }
};

inline void MPI_Exchange(void * out, int out_count, void * in, int in_count, MPI_Datatype datatype, MPI_Comm intercomm, MPI_Comm comm) {
   MPI_Request request;
   MPI_Status status;
   int rank;
   MPI_Comm_rank(intercomm, &rank);
   fflush(stdout);
   if (rank == 0) {
      MPI_Isend(out, out_count, datatype, 0, 123, intercomm, &request);
      MPI_Recv(in, in_count, datatype, 0, 123, intercomm, &status);
      MPI_Wait(&request,  &status);
   }
   MPI_Bcast(in, in_count, datatype, 0, comm);
}

template <typename T> inline MPI_Datatype MPI_Auto_datatype();
template <> inline MPI_Datatype MPI_Auto_datatype< int >() { return MPI_INT; }
template <> inline MPI_Datatype MPI_Auto_datatype< unsigned int >() { return MPI_UNSIGNED; }
template <> inline MPI_Datatype MPI_Auto_datatype< long int >() { return MPI_LONG; }
template <> inline MPI_Datatype MPI_Auto_datatype< unsigned long int >() { return MPI_UNSIGNED_LONG; }
template <> inline MPI_Datatype MPI_Auto_datatype< char >() { return MPI_CHAR; }

template <class T> inline T MPI_Exchange(T& out, MPI_Comm intercomm, MPI_Comm comm) {
   T ret;
   MPI_Exchange(&out, 1, &ret, 1, MPI_Auto_datatype<T>(), intercomm, comm);
   return ret;
};

template <>
inline int MPI_Exchange(int& out, MPI_Comm intercomm, MPI_Comm comm) {
   int ret;
   MPI_Exchange(&out, 1, &ret, 1, MPI_INT, intercomm, comm);
   return ret;
}

template <class T>
inline T MPI_Exchange_container(T& out, MPI_Comm intercomm, MPI_Comm comm) {
   size_t my_size = out.size();
   size_t other_size = MPI_Exchange(my_size, intercomm, comm);
   size_t mx = my_size;
   if (other_size > mx) mx = other_size;
   T ret;
   for (size_t i=0; i<mx; i++) {
      typename T::value_type a,b;
      if (i < my_size) a = out[i];
      b = MPI_Exchange(a, intercomm, comm);
      if (i < other_size) ret.push_back(b);
   }
   return ret;
}

template <class T>
inline std::vector<T> MPI_Exchange(std::vector<T>& out, MPI_Comm intercomm, MPI_Comm comm) {
   return MPI_Exchange_container(out, intercomm, comm);
}

template <>
inline std::string MPI_Exchange(std::string& out, MPI_Comm intercomm, MPI_Comm comm) {
   return MPI_Exchange_container(out, intercomm, comm);
}


class MPMDHelper {
   static char * char_vec_from_string( std::string str, size_t size) {
      char * ret = new char[size];
      size_t i=0;
      for (; i<str.size(); i++) ret[i] = str[i];
      for (; i<size; i++) ret[i] = '\0';
      return ret;
   }


public:
   MPI_Comm world;
   MPI_Comm local;
   MPI_Group local_group;
   MPI_Comm work;
   MPI_Group work_group;
   
   int world_rank, world_size;
   int local_rank, local_size;
   int work_rank,  work_size;
   int universe_size;
   bool in_work;
   typedef MPMDIntercomm intercomm_t;
   typedef std::map< std::string, intercomm_t > intercomm_map;
   intercomm_map intercomm;
   intercomm_t parent;

   int leader;
   int local_leader;
   
   std::string name;
   std::string host_name;

   inline MPMDHelper() {
      world = MPI_COMM_NULL;
      local = MPI_COMM_NULL;
      local_rank = -1;
   };

   inline ~MPMDHelper() {
   };

   MPMDIntercomm& operator[](const std::string& key) {
      return intercomm[key];
   };

   
   inline void Init(const MPI_Comm& world_, const std::string& name_, std::vector<int> excl = std::vector<int>()) {
      int mylen,maxlen;
      
      world = world_;
      
      std::vector<std::string> names;
      std::vector<int> leaders;
      int color = -1;
      int colors = 0;
      
      name = name_;
      MPI_Comm_rank(world, &world_rank);
      MPI_Comm_size(world, &world_size);
      {
         char host_name_c[MPI_MAX_PROCESSOR_NAME];
         int host_len;
         MPI_Get_processor_name( host_name_c, &host_len );
         host_name = host_name_c;
      }

      {
         int *universe_sizep, flag;
         MPI_Attr_get(MPI_COMM_WORLD, MPI_UNIVERSE_SIZE, &universe_sizep, &flag);  
         if (!flag) { 
           universe_size = 0; // LCOV_EXCL_LINE
         } else universe_size = *universe_sizep;
      }

      mylen = name.size() + 1;
      MPI_Allreduce( &mylen, &maxlen, 1, MPI_INT, MPI_MAX, world);
      MPI_Barrier(world);
      char* my_name = char_vec_from_string(name, maxlen);
      char* other_name = char_vec_from_string("", maxlen);

      
      color = -1;
      colors = 0;
      int rank, who;
      rank = world_rank;
      int end_value = world_size;
      while (colors <= world_size) {
         MPI_Allreduce( &rank, &who, 1, MPI_INT, MPI_MIN, world);
         if (who == end_value) break;
         if (world_rank == who) {
            for (int i=0; i<maxlen; i++) other_name[i] = my_name[i];
         }
         MPI_Bcast(other_name, maxlen, MPI_CHAR, who, world);
         names.push_back(other_name);
         leaders.push_back(who);
         if (strcmp(my_name, other_name) == 0) {
            rank = end_value;
            color = colors;
            leader = who;
         }
         colors++;
      }
      assert(colors <= world_size);
      delete[] my_name;
      delete[] other_name;
      MPI_Comm_split(world, color, rank, &local);
      MPI_Comm_rank(local, &local_rank);
      MPI_Comm_size(local, &local_size);

      MPI_Comm_group(local,&local_group);
      if (excl.size() > 0) {
         MPI_Group_excl(local_group,excl.size(),&excl[0],&work_group);
         MPI_Group_rank(work_group, &work_rank);
         MPI_Group_size(work_group, &work_size);
         MPI_Comm_create(local,work_group,&work);
         in_work = (work_rank != MPI_UNDEFINED);
      } else {
         work_group=local_group;
         work = local;
         work_rank = local_rank;
         work_size = local_size;
         in_work = true;
      }
      
      for (int c=0; c<colors; c++) {
         rank = local_rank;
         MPI_Bcast(&rank, 1, MPI_INT, leaders[c], world);
         if (color == c) local_leader = rank;
      }
      for (int c=0; c<colors; c++) {
         MPI_Comm inter = MPI_COMM_NULL;
         if (c != color) {
            int tag;
            if (c < color) {
               tag = c*colors + color;
            } else {
               tag = color*colors + c;
            }
            MPI_Intercomm_create(local, local_leader, world, leaders[c], tag, &inter);
            ConnectIntercomm(inter, true);
         }
      }
      
      MPI_Comm parent_;
      MPI_Comm_get_parent(&parent_);
      if (parent_ != MPI_COMM_NULL) {
         parent=ConnectIntercomm(parent_, false);
      }
      
   }


   inline intercomm_t ConnectIntercomm(MPI_Comm inter, bool world_=true, bool work_=false) {
      intercomm_t ret;
      ret.in_world = world_;
      MPI_Comm my_comm = local;
      if (work_) my_comm = work;
      int my_work = work_ && (local != work);
      int other_work;
//      MPI_Exchange(&my_work, 1, &other_work, 1, MPI_INT, inter, my_comm);
      other_work = MPI_Exchange(my_work, inter, my_comm);
      ret.in_local = ! (my_work || other_work);

      ret.name = MPI_Exchange(name, inter, my_comm);
      
      if (ret.in_local) {
         ret.local = inter;
         MPI_Comm_remote_size(ret.local, &ret.local_size);
      } else {         
         ret.local = MPI_COMM_NULL;
         ret.local_size = 0;
      }
      
      int my_trivial_group = (my_comm == work);
      int other_trivial_group;
      MPI_Exchange(&my_trivial_group, 1, &other_trivial_group, 1, MPI_INT, inter, my_comm);
      if (my_trivial_group && other_trivial_group) {
         ret.work = inter;
      } else {
         MPI_Comm_create(inter, work_group, &ret.work);
      }

      if (ret.work != MPI_COMM_NULL) {
         ret.in_work = true;
         MPI_Comm_remote_size(ret.work, &ret.work_size);
      } else {
         ret.in_work = false;
         ret.work_size = 0;
      }
      ret.connected = true;
      intercomm.insert(make_pair(ret.name, ret));
      return ret;
   }

   inline intercomm_t Spawn( char *command, char *argv[], int maxprocs, MPI_Info info, bool work_ = false) {
      MPI_Comm comm, inter;
      std::vector<int> err(maxprocs);
      int root;
      if (work_) {
         comm = work;
         root = 0;
      } else {
         comm = local;
         root = 0;
      }
      MPI_Comm_spawn( command, argv, maxprocs, info, root, comm, &inter, &err[0]);
      if (inter != MPI_COMM_NULL) {
         return ConnectIntercomm(inter, false, work_);
      } else {
         intercomm_t ret;
         return ret;
      }
   }

   inline void Identify(bool world_ = false) {
//      printf("MPMD: %03d.%03d world:%d/%d local:%d/%d color:%d/%d leader:%d/%d --- %s @ %s\n", color, local_rank, world_rank, world_size, local_rank, local_size,  color, colors, local_leader, leader, my_name, host_name);
      char iden_c[400];
      std::string iden = "MPMD: ";
      iden += name;
      iden += ":";
      if (world_) {
         sprintf(iden_c, " world:%d/%d", world_rank, world_size); iden += iden_c; // LCOV_EXCL_LINE
      }
      sprintf(iden_c, " local:%d/%d", local_rank, local_size); iden += iden_c;
      if (in_work) {
         sprintf(iden_c, " work:%d/%d", work_rank, work_size);
      } else {
         sprintf(iden_c, " work:NA/%d", work_size);
      }      
      iden += iden_c;
      iden += " --- ";
      iden += " connected to:";
      for (intercomm_map::iterator c = intercomm.begin(); c != intercomm.end(); c++) {
         intercomm_t inter = c->second;
         iden += " ";
         iden += inter.name;
         iden += "(";
         if (inter.in_work) { sprintf(iden_c, "%d", inter.work_size); iden += iden_c; } else iden += "NA";
         iden += "/";
         if (inter.in_local) { sprintf(iden_c, "%d", inter.local_size); iden += iden_c; } else iden += "NA";
         iden += ")";
      }
      
      printf("%s\n",iden.c_str());
   }
};

