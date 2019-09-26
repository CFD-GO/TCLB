#include "MPMD.hpp"
#include <lammps/lammps.h>
#include <lammps/library.h>
#include "RemoteForceInterface.hpp"

int main(int argc, char *argv[])
{
   int ret;
   MPMDHelper MPMD;
   MPI_Init(&argc, &argv);
   MPMD.Init(MPI_COMM_WORLD, "LAMMPS");
   MPMD.Identify();

   rfi::RemoteForceInterface< rfi::ForceCalculator, rfi::RotParticle > RFI;
   
   MPMDIntercomm inter = MPMD["TCLB"];
   if (!inter) {
     fprintf(stderr,"Didn't find TCLB in MPMD\n");
     return -1;
   }

   ret = RFI.Connect(MPMD.work,inter.work);
   if (ret) return ret;
   assert(RFI.Connected());

   RFI.Close();
   MPI_Finalize();
   return 0;
}
