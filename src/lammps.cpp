#include "MPMD.hpp"
#include <lammps/lammps.h>
#include <lammps/library.h>
#include <lammps/input.h>
#include <lammps/modify.h>
#include <lammps/atom.h>
#include <lammps/fix.h>
#include <lammps/fix_external.h>
#include "RemoteForceInterface.hpp"

using namespace LAMMPS_NS;



struct Info {
   MPMDHelper *MPMD;
   Memory *memory;
   LAMMPS *lmp;
   rfi::RemoteForceInterface< rfi::ForceIntegrator, rfi::RotParticle > * RFI;
};


void quest_callback(void *, bigint, int, int *, double **, double **);

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
   MPMDHelper MPMD;
   MPI_Init(&argc, &argv);
   MPMD.Init(MPI_COMM_WORLD, "LAMMPS");
   MPMD.Identify();
   rfi::RemoteForceInterface< rfi::ForceIntegrator, rfi::RotParticle > RFI;
   RFI.name = "LAMMPS";
   MPMDIntercomm inter = MPMD["TCLB"];
   if (!inter) {
     fprintf(stderr,"Didn't find TCLB in MPMD\n");
     return -1;
   }
   ret = RFI.Connect(MPMD.work,inter.work);
   if (ret) return ret;
   assert(RFI.Connected());

   if (argc != 2) {
     printf("Syntax: lammps in.lammps\n");
     MPI_Abort(MPI_COMM_WORLD,1);
     exit(1);
   }
   FILE *fp;
   if (MPMD.local_rank == 0) {
     fp = fopen(argv[1],"r");
     if (fp == NULL) {
       printf("ERROR: Could not open LAMMPS input script\n");
       MPI_Abort(MPI_COMM_WORLD,1);
       exit(1);
     }
   }

   LAMMPS *lmp = NULL;
   lmp = new LAMMPS(0,NULL,MPMD.local);

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
      printf("LAMMPS> %s",line);
     }
     
     lammps_command(lmp,line);
     
     if (match_pattern(line, " fix tclb * external")) {
       printf("LAMMPS: Added fix tclb external! (%s) Adding callback.\n", line);
       int ifix = lmp->modify->find_fix("tclb");
       printf("ifix: %d\n",ifix);
       FixExternal *fix = (FixExternal *) lmp->modify->fix[ifix];
       printf("fix: %p\n",fix);
       
       Info info;
       info.MPMD = &MPMD;
       info.memory = NULL;
       info.lmp = lmp;
       info.RFI = &RFI;

       fix->set_callback(quest_callback,&info);
     }
   }
   
   lammps_close(lmp);
   RFI.Close();
   RFI.Close();
   MPI_Finalize();
   return 0;
}


void quest_callback(void *ptr, bigint ntimestep, int nlocal, int *id, double **x, double **f)
{
   Info *info = (Info *) ptr;
   if (! info->RFI->Active()) return;
   for (int j = 0; j < info->RFI->Workers(); j++) {
     info->RFI->Size(j) = nlocal;
   }
   
   info->RFI->Alloc();
   int i=0;
   double ** v = info->lmp->atom->v;
   double * r = info->lmp->atom->radius;
   for (int j = 0; j < info->RFI->Workers(); j++) {
    for (size_t k = 0; k < nlocal; k++) {
     info->RFI->setData(i, RFI_DATA_R,      r[k]);
     info->RFI->setData(i, RFI_DATA_POS+0,  x[k][0]);
     info->RFI->setData(i, RFI_DATA_POS+1,  x[k][1]);
     info->RFI->setData(i, RFI_DATA_POS+2,  x[k][2]);
     info->RFI->setData(i, RFI_DATA_VEL+0,  v[k][0]);
     info->RFI->setData(i, RFI_DATA_VEL+1,  v[k][1]);
     info->RFI->setData(i, RFI_DATA_VEL+2,  v[k][2]);
     if (info->RFI->Rot()) {
       info->RFI->setData(i, RFI_DATA_ANGVEL+0,  0.0);
       info->RFI->setData(i, RFI_DATA_ANGVEL+1,  0.0);
       info->RFI->setData(i, RFI_DATA_ANGVEL+2,  0.0);
     }
     i++;
    }
   }
   
   info->RFI->SendSizes();
   info->RFI->SendParticles();
   info->RFI->SendForces();
   for (size_t k = 0; k < nlocal; k++) {
    f[k][0] = 0;
    f[k][1] = 0;
    f[k][2] = 0;
   }
   i=0;
   for (int j = 0; j < info->RFI->Workers(); j++) {
    for (size_t k = 0; k < nlocal; k++) {
     f[k][0] += info->RFI->getData(i, RFI_DATA_FORCE+0);
     f[k][1] += info->RFI->getData(i, RFI_DATA_FORCE+1);
     f[k][2] += info->RFI->getData(i, RFI_DATA_FORCE+2);
     i++;
    }
   }
}

