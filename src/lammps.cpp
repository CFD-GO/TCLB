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

   double di = 100.0 * MPMD.work_rank;
   for (int i = 0; i < RFI.Workers(); i++) {
     RFI.Size(i) = 3;
   }
   RFI.Alloc();
   for (size_t i = 0; i < RFI.size(); i++) {
     RFI.setData(i, RFI_DATA_R,      di+i+0.001);
     RFI.setData(i, RFI_DATA_POS+0,  di+i+0.002);
     RFI.setData(i, RFI_DATA_POS+1,  di+i+0.003);
     RFI.setData(i, RFI_DATA_POS+2,  di+i+0.004);
     RFI.setData(i, RFI_DATA_VEL+0,  di+i+0.005);
     RFI.setData(i, RFI_DATA_VEL+1,  di+i+0.006);
     RFI.setData(i, RFI_DATA_VEL+2,  di+i+0.007);
     if (RFI.Rot()) {
       RFI.setData(i, RFI_DATA_ANGVEL+0,  i+0.008);
       RFI.setData(i, RFI_DATA_ANGVEL+1,  i+0.009);
       RFI.setData(i, RFI_DATA_ANGVEL+2,  i+0.010);
     }
   }
   
   while ( RFI.Active() ) {
       printf("LAMMPS: RFI.SendSizes()\n");
       RFI.SendSizes();
       printf("RFI size sent ...\n");
       RFI.SendParticles();
       printf("RFI particles send ...\n");
       RFI.SendForces();
       printf("RFI forces send ...\n");
   }
   
   RFI.Close();


   RFI.Close();
   MPI_Finalize();
   return 0;
}
