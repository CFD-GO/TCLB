#include "acRemoteForceInterface.h"
std::string acRemoteForceInterface::xmlname = "RemoteForceInterface";
#include "../HandlerFactory.h"


int acRemoteForceInterface::Init () {

        int workers = solver->lattice->RFI.space_for_workers() - 1;
        if (workers < 1) {
                ERROR("ESYS-P: No place for workers (you need at least 2 additionals processes)\n");
                return -1;
        }
        int nx=1, ny=1, nz=1;
        int nx0, ny0, nz0;
        int nx1, ny1, nz1;
        int tot, tot1;
        tot=0;
        nx0 = solver->mpi.divx;
        ny0 = solver->mpi.divy;
        nz0 = solver->mpi.divz;
        output("%dx%dx%d\n",nx0, ny0, nz0);
        for (int nx1=1; nx1<=nx0; nx1++) if (nx0 % nx1 == 0) {
                for (int ny1=1; ny1<=ny0; ny1++) if (ny0 % ny1 == 0) {
                        for (int nz1=1; nz1<=nz0; nz1++) if (nz0 % nz1 == 0) {
                                int tot1 = nx1*ny1*nz1;
                                if (tot1 <= workers) {
                                        if (workers % tot1 == 0) {
                                                if (tot1 > tot) {
                                                        tot = tot1;
                                                        nx = workers / (ny1 * nz1);
                                                        ny = ny1;
                                                        nz = nz1;
                                                }
                                        }
                                }
                        }
                }
        }
        if (tot == 0) {
                ERROR("ESYS-P: Cannot find a good division. Requested workers (%d) do not fit well with TCLB division (%dx%dx%d)\n", workers, nx0, ny0, nz0);
                return -1;
        }
        output("ESYS-P: Will be running at %dx%dx%d\n", nx, ny, nz);
        
                        char fn[STRING_LEN];
//            		solver->outGlobalFile("ESYS", ".py", fn);
                        sprintf(fn, "%s_%s.py", solver->info.outpath, "ESYS");
                        output("ESYS-P: config: %s\n", fn);
                        if (D_MPI_RANK == 0) {
                                FILE * f = fopen(fn, "wt");
                                fprintf(f, "from esys.lsm import *\n");
                                fprintf(f, "from esys.lsm.util import Vec3, BoundingBox\n");
                                fprintf(f, "from esys.lsm.geometry import *\n\n");
                                fprintf(f, "sim = LsmMpi(numWorkerProcesses=%d, mpiDimList=[%d,%d,%d])\n",nx*ny*nz,nx,ny,nz);
                                fprintf(f, "sim.initNeighbourSearch( particleType=\"%s\",	gridSpacing=%lg, verletDist=%lg )\n", "NRotSphere", 25.0, 5.0);
                                fprintf(f, "sim.setSpatialDomain( BoundingBox(Vec3(%lg,%lg,%lg), Vec3(%lg,%lg,%lg)), circDimList = [%s, False, False])\n", 0.0, 0.0, 0.0, 320.0, 128.0, 0.0, "True");
                                fprintf(f, "sim.setTimeStepSize(%lg)\n", 1.0);
                                fprintf(f, "sim.setNumTimeSteps(%d)\n",(int) 90000);
                                fprintf(f, "sim.createInteractionGroup(	TCLBForcePrms(name=\"tclb\", acceleration=Vec3(0,-9.81,0), fluidDensity=1.0, fluidHeight=0) )\n");
                                fprintf(f, "%s\n", node.child_value());
                                fprintf(f, "sim.run()\n");
                                fflush(f);
                                fclose(f);
                        }
                        MPI_Barrier(MPI_COMM_WORLD);

        char * args[] = {fn,NULL};
        
        solver->lattice->RFI.Start("esysparticle",args);
	return 0;
}


// Register the handler (basing on xmlname) in the Handler Factory
template class HandlerFactory::Register< GenericAsk< acRemoteForceInterface > >;
