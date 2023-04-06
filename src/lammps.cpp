#include "MPMD.hpp"
#include <lammps/lammps.h>
#include <lammps/domain.h>
#include <lammps/library.h>
#include <lammps/input.h>
#include <lammps/modify.h>
#include <lammps/atom.h>
#include <lammps/fix.h>
#include <lammps/fix_external.h>
#include "Global.h"
#include "RemoteForceInterface.hpp"
#include <vector>

using namespace LAMMPS_NS;


typedef rfi::RemoteForceInterface< rfi::ForceIntegrator, rfi::RotParticle, rfi::ArrayOfStructures, real_t > RFI_t;

struct Info {
   MPMDHelper *MPMD;
   Memory *memory;
   LAMMPS *lmp;
   RFI_t * RFI;
   std::vector<size_t> wsize;
   std::vector<size_t> windex;
};


void tclb_callback(void *, bigint, int, int *, double **, double **);

int match_pattern(char* str, char* pattern) {
       bool fit = true;
       while (fit) {
          if (pattern[0] == '\0') break;
          if (pattern[0] == ' ') {
             if (str[0] == ' ') str++;
             else if (str[0] == '\t') str++;
             else pattern++;
          } else if (pattern[0] == '*') {
             if (str[0] == ' ') pattern++;
             else if (str[0] == '\t') pattern++;
             else str++;
          } else if (pattern[0] == str[0]) {
             pattern++; str++;
          } else fit = false;
       }
       return fit;
}


int main(int argc, char *argv[])
{
   int ret;
   Info info;
   MPMDHelper MPMD;
   MPI_Init(&argc, &argv);
   MPMD.Init(MPI_COMM_WORLD, "LAMMPS");
   DEBUG_SETRANK(MPMD.local_rank);
   InitPrint(DEBUG_LEVEL, 6, 8);
   MPMD.Identify();
   RFI_t RFI;
   RFI.name = "LAMMPS";

   if (argc < 2) {
     printf("Syntax: lammps in.lammps [args]\n");
     MPI_Abort(MPI_COMM_WORLD,1);
     exit(1);
   }
   std::vector<char*> lammps_args;
   char * infile = NULL;
   bool logset = false;
   for (int i=0; i<argc; i++) {
     if ((i == 1) && (argv[i][0] != '-')) {
       infile = argv[i];
     } else if (strcmp(argv[i],"-in") == 0) {
       i++;
       if (i < argc) {
         infile = argv[i];
       } else {
         printf("ERROR: No filename after '-in'\n");
         MPI_Abort(MPI_COMM_WORLD,1);
         exit(1);
       }
     } else {
       if (strcmp(argv[i],"-log") == 0) logset = true;
       lammps_args.push_back(argv[i]);
     }
   }
   if (!logset) {
     printf("LAMMPS: notice: switching off the default log\n");
     lammps_args.push_back("-log");
     lammps_args.push_back("none");
   }

   if (infile == NULL) {
     printf("ERROR: No filename provided\n");
     MPI_Abort(MPI_COMM_WORLD,1);
     exit(1);
   }
   
   FILE *fp = NULL;
   if (MPMD.local_rank == 0) {
     printf("LAMMPS: running input file: %s\n", infile);
     printf("LAMMPS: arguments:");
     for (int i=1; i<lammps_args.size(); i++) printf(" %s", lammps_args[i]);
     printf("\n");
     fp = fopen(infile,"r");
     if (fp == NULL) {
       printf("ERROR: Could not open LAMMPS input script %s\n", infile);
       MPI_Abort(MPI_COMM_WORLD,1);
       exit(1);
     }
   }

   LAMMPS *lmp = NULL;
   lmp = new LAMMPS(lammps_args.size(),&lammps_args[0],MPMD.local);

   int n;
   char line[1024];
   while (1) {
     if (MPMD.local_rank == 0) {
       if (fgets(line,1024,fp) == NULL) n = 0;
       else n = strlen(line) + 1;
       if (n == 0) fclose(fp);
     }
     MPI_Bcast(&n,1,MPI_INT,0,MPMD.local);
     if (n == 0) break;
     MPI_Bcast(line,n,MPI_CHAR,0,MPMD.local);
     if (MPMD.local_rank == 0) {
      fprintf(stdout,"LAMMPS> %s",line); fflush(stdout);
     }

     lammps_command(lmp,line);

     if (match_pattern(line, " fix tclb * external")) {
       printf("LAMMPS: Added fix tclb external! (%s) Adding callback.\n", line);
       MPMDIntercomm inter = MPMD["TCLB"];
       if (!inter) {
         fprintf(stderr,"Didn't find TCLB in MPMD\n");
         return -1;
       }
       ret = RFI.Connect(MPMD.work,inter.work);
       if (ret) return ret;
       assert(RFI.Connected());

       int ifix = lmp->modify->find_fix("tclb");
       printf("ifix: %d\n",ifix);
       FixExternal *fix = (FixExternal *) lmp->modify->fix[ifix];
       printf("fix: %p\n",fix);

       info.MPMD = &MPMD;
       info.memory = NULL;
       info.lmp = lmp;
       info.RFI = &RFI;
       info.wsize.resize(RFI.Workers());
       info.windex.resize(RFI.Workers());

       fix->set_callback(tclb_callback,&info);
     }
   }

   lammps_close(lmp);
   if (RFI.Connected()) {
     RFI.Close();
   }
   MPI_Finalize();
   return 0;
}


void tclb_callback(void *ptr, bigint ntimestep, int nlocal, int *id, double **x_, double **f)
{
   Info *info = (Info *) ptr;
   if (! info->RFI->Active()) return;

   double ** v = info->lmp->atom->v;
   double ** x = info->lmp->atom->x;
   double * r = info->lmp->atom->radius;

   for (int phase = 0; phase < 3; phase++) {
    if (phase == 0) {
     for (int i = 0; i < info->RFI->Workers(); i++) info->wsize[i] = 0;
    } else {
     for (int i = 0; i < info->RFI->Workers(); i++) info->windex[i] = 0;
    }

    for (size_t k = 0; k < nlocal; k++) {
     if (phase == 2) {
      f[k][0] = 0;
      f[k][1] = 0;
      f[k][2] = 0;
     }
     int minper[3], maxper[3], d[3];
     size_t offset = 0;
     for (int worker = 0; worker < info->RFI->Workers(); worker++) {
      for (int j=0; j<3; j++) {
       double prd = info->lmp->domain->prd[j];
       double lower = 0;
       double upper = info->lmp->domain->prd[j];
       if (info->RFI->WorkerBox(worker).declared) {
         lower = info->RFI->WorkerBox(worker).lower[j];
         upper = info->RFI->WorkerBox(worker).upper[j];
       }
       if (info->lmp->domain->periodicity[j]) {
         maxper[j] = floor((upper - x[k][j] + r[k]) / prd);
         minper[j] =  ceil((lower - x[k][j] - r[k]) / prd);
       } else {
         if ((x[k][j] + r[k] >= lower) && (x[k][j] - r[k] <= upper)) {
           minper[j] = 0; maxper[j] = 0;
         } else {
           minper[j] = 0; maxper[j] = -1; // no balls
         }
       }
//       printf("particle %ld dimenstion %d in %d worker interval [%lg %lg] and periodicity %lg: %lg copied %d:%d\n", k, j, worker, lower, upper, prd, x[k][j], minper[j], maxper[j]);
      }
      
      int copies = (maxper[0]-minper[0]+1)*(maxper[1]-minper[1]+1)*(maxper[2]-minper[2]+1);
 //     if (copies > 1) printf("particle %ld is copied %d times (%d %d)x(%d %d)x(%d %d)\n", k, copies, minper[0], maxper[0], minper[1], maxper[1], minper[2], maxper[2]);
      for (d[0]=minper[0]; d[0]<=maxper[0]; d[0]++) {
       for (d[1]=minper[1]; d[1]<=maxper[1]; d[1]++) {
        for (d[2]=minper[2]; d[2]<=maxper[2]; d[2]++) {
         double px[3];
         for (int j=0; j<3; j++) px[j] = x[k][j] + d[j] * info->lmp->domain->prd[j];
         if (phase == 0) {
          info->wsize[worker]++;
         } else {
          size_t i = offset + info->windex[worker];
          if (phase == 1) {
           //printf("particle %ld sent %d at index %ld\n", k, worker, i);
           info->RFI->setData(i, RFI_DATA_R,      r[k]);
           info->RFI->setData(i, RFI_DATA_POS+0,  px[0]);
           info->RFI->setData(i, RFI_DATA_POS+1,  px[1]);
           info->RFI->setData(i, RFI_DATA_POS+2,  px[2]);
           info->RFI->setData(i, RFI_DATA_VEL+0,  v[k][0]);
           info->RFI->setData(i, RFI_DATA_VEL+1,  v[k][1]);
           info->RFI->setData(i, RFI_DATA_VEL+2,  v[k][2]);
           if (info->RFI->Rot()) {
            info->RFI->setData(i, RFI_DATA_ANGVEL+0,  0.0);
            info->RFI->setData(i, RFI_DATA_ANGVEL+1,  0.0);
            info->RFI->setData(i, RFI_DATA_ANGVEL+2,  0.0);
           }
          } else {
           f[k][0] += info->RFI->getData(i, RFI_DATA_FORCE+0);
           f[k][1] += info->RFI->getData(i, RFI_DATA_FORCE+1);
           f[k][2] += info->RFI->getData(i, RFI_DATA_FORCE+2);
          }
          info->windex[worker]++;
         }
        }
       }
      }
      offset += info->wsize[worker];
     }
    }
    if (phase == 0) {
     for (int worker = 0; worker < info->RFI->Workers(); worker++) info->RFI->Size(worker) = info->wsize[worker];
     //printf("sizes:"); for (int worker = 0; worker < info->RFI->Workers(); worker++) printf(" %ld", (size_t) info->wsize[worker]); printf("\n");
     info->RFI->SendSizes();
     info->RFI->Alloc();
    } else if (phase == 1) {
     //printf("indexes:"); for (int worker = 0; worker < info->RFI->Workers(); worker++) printf(" %ld", (size_t) info->windex[worker]); printf("\n");
     info->RFI->SendParticles();
     info->RFI->SendForces();
    } else {
    }
   }
}

