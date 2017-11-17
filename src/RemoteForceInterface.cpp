/* manager */ 
#include "RemoteForceInterface.h"
#include "Global.h"

RemoteForceInterface::RemoteForceInterface() : workers(0), masters(0), intercomm(MPI_COMM_NULL), totsize(0) {
   int *universe_sizep, flag;
   MPI_Comm_size(MPI_COMM_WORLD, &world_size); 
   MPI_Attr_get(MPI_COMM_WORLD, MPI_UNIVERSE_SIZE, &universe_sizep, &flag);  
   if (!flag) { 
     universe_size = 0;
   } else universe_size = *universe_sizep;
   sent = false;
}

RemoteForceInterface::~RemoteForceInterface() {
  if (intercomm != MPI_COMM_NULL) {
    MPI_Comm_free(&intercomm);
  }
}

int RemoteForceInterface::Start(char * worker_program, char * args[], double units[]) {
   if (intercomm != MPI_COMM_NULL) {
    error("RemoteForceInterface(M) Already started\n");
    return -2;
   }
   if (universe_size == world_size) {
    ERROR("No room to start workers"); 
    return -1;
   }
   MPI_Comm everyone;
   MPI_Comm_spawn(worker_program, args, universe_size - world_size,
             MPI_INFO_NULL, 0, MPI_COMM_WORLD, &everyone,  
             MPI_ERRCODES_IGNORE); 
             
   MPI_Group group;
   MPI_Comm_group(MPI_COMM_WORLD, &group);
   MPI_Comm_create(everyone, group, &intercomm);
   
   {
    int s1,s2;
    MPI_Comm_remote_size(everyone, &s1);
    MPI_Comm_remote_size(intercomm, &s2);
    output("RemoteForceInterface(M) Total children: %d, intercomm with: %d\n",s1,s2);
   }
   MPI_Comm_remote_size(intercomm, &workers);
   MPI_Comm_size(intercomm, &masters);
   MPI_Comm_rank(intercomm, &rank);
   sizes.resize(workers, 0);
   nsizes.resize(workers, 0);
   offsets.resize(workers+1, 0);
   reqs.resize(workers+1);
   stats.resize(workers+1);
   int root = MPI_PROC_NULL; if (rank == 0) root = MPI_ROOT;
   MPI_Bcast(&units[0], 1, MPI_DOUBLE, root, intercomm);
   MPI_Bcast(&units[1], 1, MPI_DOUBLE, root, intercomm);
   MPI_Bcast(&units[2], 1, MPI_DOUBLE, root, intercomm);
   return 0;
}

void RemoteForceInterface::Close() {
  if (intercomm == MPI_COMM_NULL) return;
  output("RemoteForceInterface(M) Closing ...\n");
  totsize = 0;
  for (int i=0; i<workers; i++) {
   sizes[i] = 0;
   offsets[i] = 0;
  }
  MPI_Comm_free(&intercomm);
  intercomm = MPI_COMM_NULL;
}


void RemoteForceInterface::GetSizes() {
    if (intercomm == MPI_COMM_NULL) return;
    if (sent) {
     debug1("Wait for it ... (GetSizes)\n");
     MPI_Waitall(workers+1, &reqs[0], &stats[0]);
//     printf("Wait for it ... (GetSizes2)\n");
//     MPI_Ialltoall(NULL, 0, MPI_RFI_SIZE_T, &sizes[0], 1, MPI_RFI_SIZE_T, intercomm, &reqs[workers]);
//     MPI_Waitall(1, &reqs[workers], &stats[workers]);
     for (int i=0; i<workers; i++) sizes[i] = nsizes[i];
     sent = false;
    } else {
         debug1("RemoteForceInterface(M) Exchange of sizes ...\n");
         MPI_Request req;
         MPI_Status stat;
         MPI_Ialltoall(NULL, 0, MPI_RFI_SIZE_T, &sizes[0], 1, MPI_RFI_SIZE_T, intercomm, &req);
         MPI_Wait(&req, &stat);
    }
    for (int i=0; i<workers; i++) if (sizes[i] == RFI_FINISHED) { Close(); return; }
    for (int i=0; i<workers; i++) debug1("RemoteForceInterface(M) [%2d] we got %ld from %d\n", rank, (size_t) sizes[i], i);
    for (int i=0; i<workers; i++) offsets[i+1] = offsets[i] + sizes[i];
    totsize = offsets[workers];
    if (totsize > tab.size()) tab.resize(totsize);
    debug1("RemoteForceInterface(M) Sending ...\n");
    for (int i=0; i<workers; i++) {
        MPI_Irecv(&tab[offsets[i]], sizes[i], MPI_RFI_REAL_T, i, i, intercomm, &reqs[i]);
    }
}

void RemoteForceInterface::GetParticles() {
    if (intercomm == MPI_COMM_NULL) return;
    debug1("Wait for it ... (GetParticles)\n");
    MPI_Waitall(workers, &reqs[0], &stats[0]);
//    intercomm = MPI_COMM_NULL;
}

void RemoteForceInterface::SetParticles() {
    if (intercomm == MPI_COMM_NULL) return;
    debug1("RemoteForceInterface(M) Receiving ...\n");
    for (int i=0; i<workers; i++) {
        MPI_Isend(&tab[offsets[i]], sizes[i], MPI_RFI_REAL_T, i, i+workers, intercomm, &reqs[i]);
    }
    MPI_Ialltoall(NULL, 0, MPI_RFI_SIZE_T, &nsizes[0], 1, MPI_RFI_SIZE_T, intercomm, &reqs[workers]);
    sent=true;
}

