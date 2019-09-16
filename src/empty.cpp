#include "MPMD.hpp"

int main(int argc, char *argv[])
{
   MPMDHelper MPMD;
   MPI_Init(&argc, &argv);
   MPMD.Init(MPI_COMM_WORLD, "NOTHING");
   MPMD.Identify();
   MPI_Finalize();
   return 0;
}
