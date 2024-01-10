#include "acESYSParticle.h"
std::string acESYSParticle::xmlname = "ESYSParticle";
#include "../HandlerFactory.h"


int acESYSParticle::Init () {
        Action::Init();
        bool need_file = true;
        MPMDIntercomm inter_;
        pugi::xml_attribute attr;
        char fn[2*STRING_LEN];
        std::string filename;
        int N;
        double units[3];
        units[0] = solver->units.alt("1m");
        units[1] = solver->units.alt("1s");
        units[2] = solver->units.alt("1kg");

        printf("uni:%d - world:%d\n",MPMD.universe_size, MPMD.world_size);
        N = MPMD.universe_size - MPMD.world_size;

        inter_ = MPMD["ESYSPARTICLE"];
        if (inter_) need_file = false;

        if (need_file) {
                attr = node.attribute("script");
                if (attr) {
                        sprintf(fn,"%s",attr.value());
                        filename = attr.value();
                        need_file = false;
                }
        }
        if (need_file) {
                xcirc = false;
                ycirc = false; //JM
                zcirc = false; //JM
                particle_type = "NRotSphere";
                sim = "sim";
                gridSpacing = 25.0;
                verletDist = 5.0;
                double sx,sy,sz; // Size of the domain for ESYS
                attr = node.attribute("particle");
                if (attr) particle_type = attr.value();
                if (particle_type == "NRotSphere") {
                } else if (particle_type == "RotSphere") {
                } else {
                        ERROR("Unknown particle type in ESYS\n");
                        return -1;
                }
                attr = node.attribute("gridSpacing");
                if (attr) gridSpacing = attr.as_double();
                attr = node.attribute("verletDist");
                if (attr) verletDist = attr.as_double();
                attr = node.attribute("esys-object");
                if (attr) sim = attr.value();
                
                /*
                attr = node.attribute("periodic");
                if (attr) {
                        if (strcmp(attr.value(),"x") == 0) {
                                xcirc = true;
                        } else if (strcmp(attr.value(),"") == 0) {
                                xcirc = false;
                        } else {
                                ERROR("ESYS-Particles can be only periodic in X direction\n");
                                return -1;
                        }
                } */
                //JM Version for full periodicity
                attr = node.attribute("periodic");
                if (attr) {
                        if (strcmp(attr.value(),"x") == 0) {
                                xcirc = true;
                        } else if (strcmp(attr.value(),"y") == 0) {
                                ycirc = true;
                        } else if (strcmp(attr.value(),"z") == 0) {
                                zcirc = true;
                        } else if (strcmp(attr.value(),"x+y") == 0) {
                                xcirc = true;
                                ycirc = true;
                        } else if (strcmp(attr.value(),"x+z") == 0) {
                                xcirc = true;
                                zcirc = true;
                        } else if (strcmp(attr.value(),"y+z") == 0) {
                                ycirc = true;
                                zcirc = true;
                        } else if (strcmp(attr.value(),"x+y+z") == 0) {
                                xcirc = true;
                                ycirc = true;
                                zcirc = true;
                        } else if (strcmp(attr.value(),"") == 0) {
                                xcirc = false;
                                ycirc = false;
                                zcirc = false;
                        } else {
                                ERROR("Incorrect input options use e.g. x+y\n");
                                return -1;
                        }
                }

                const auto& global_region = solver->getCartLattice()->getGlobalRegion();
                sx = global_region.nx;
                sy = global_region.ny;
                sz = global_region.nz;
                
                int workers = N - 1;
                if (workers < 1) {
                        ERROR("ESYS-P: No place for workers (you need at least 2 additionals processes)\n");
                        return -1;
                }
                int nx=1, ny=1, nz=1;
                {
                        int nx0, ny0, nz0;
                        int tot, tot1;
                        
                        //JM 
                        int xper, yper, zper;
                        xper = xcirc ? 1 : 0;
                        yper = ycirc ? 1 : 0;
                        zper = zcirc ? 1 : 0;                        
                        
                        tot=0;
                        nx0 = solver->getCartLattice()->connectivity.divx;
                        ny0 = solver->getCartLattice()->connectivity.divy;
                        nz0 = solver->getCartLattice()->connectivity.divz;
                        if (nx0 <= xper) nx0 = xper + 1;
                        if (ny0 <= yper) ny0 = yper + 1;
                        if (nz0 <= zper) nz0 = zper + 1;
                        output("%dx%dx%d\n",nx0, ny0, nz0);
                        //JM *per additions to account for periodicity in multiple directions (hopefully ...)
                        for (int nx1=1; nx1<=nx0; nx1++) if (nx0 % nx1 == 0) {
                                for (int ny1=1; ny1<=ny0; ny1++) if (ny0 % ny1 == 0) {
                                        for (int nz1=1; nz1<=nz0; nz1++) if (nz0 % nz1 == 0) {
                                                int tot1 = nx1*ny1*nz1;
                                                if (tot1 <= workers) {
                                                        if (workers % tot1 == 0) {
                                                                int nx_ = workers / (ny1 * nz1);
                                                                if ((nx_ > xper) && (ny1 > yper) && (nz1 > zper)) {
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
                        }
                        if (tot == 0) {
                                output("Error may be due to periodicity in multiple directions calculation");
                                ERROR("ESYS-P: Cannot find a good division. Requested workers (%d) do not fit well with TCLB division (%dx%dx%d)\n", workers, nx0, ny0, nz0);
                                return -1;
                        }
                }
                output("ESYS-P: Will be running at %dx%dx%d\n", nx, ny, nz);

                if (xcirc) {
                        if (nx < 2) {
                                ERROR("ESYS-P can be periodic in X only when there are 2 processes in this direction.\n");
                                return -1;
                        }
                }
                //JM
                if (ycirc) {
                        if (ny < 2) {
                                ERROR("ESYS-P can be periodic in Y only when there are 2 processes in this direction.\n");
                                return -1;
                        }
                }
                if (zcirc) {
                        if (nz < 2) {
                                ERROR("ESYS-P can be periodic in Z only when there are 2 processes in this direction.\n");
                                return -1;
                        }
                }

                double maxRad = 10;

        //	solver->outGlobalFile("ESYS", ".py", fn);
                sprintf(fn, "%s_%s.py", solver->outpath.c_str(), "ESYS");
                output("ESYS-P: config: %s\n", fn);
                if (D_MPI_RANK == 0) {
                        FILE * f = fopen(fn, "wt");
                        fprintf(f, "from esys.lsm import *\n");
                        fprintf(f, "from esys.lsm.util import Vec3, BoundingBox\n");
                        fprintf(f, "from esys.lsm.geometry import *\n\n");
                        fprintf(f, "%s = LsmMpi(numWorkerProcesses=%d, mpiDimList=[%d,%d,%d])\n", sim.c_str(), nx*ny*nz, nx, ny, nz);
                        fprintf(f, "%s.initNeighbourSearch( particleType=\"%s\", gridSpacing=%lg, verletDist=%lg )\n", sim.c_str(), particle_type.c_str(), gridSpacing, verletDist);
                        fprintf(f, "%s.setSpatialDomain( BoundingBox(Vec3(%lg,%lg,%lg), Vec3(%lg,%lg,%lg)), circDimList = [%s, %s, %s])\n", //JM [%s,False, False]
                                sim.c_str(),
                                0.0, 0.0, 0.0,
                                sx/units[0], sy/units[0], sz/units[0],
                                xcirc ? "True" : "False",
                                ycirc ? "True" : "False",
                                zcirc ? "True" : "False");
                        fprintf(f, "%s.setTimeStepSize(%lg)\n", sim.c_str(), 1.0/units[1]);
                        fprintf(f, "%s.setNumTimeSteps(%d)\n", sim.c_str(), Next(solver->iter));
                        fprintf(f, "%s.createInteractionGroup(	RemoteForcePrms(name=\"tclb\", remote_name=\"%s\", max_rad=%lg) )\n", sim.c_str(), MPMD.name.c_str(), maxRad);
                        fprintf(f, "output_prefix=\"%s_%s\"\n", solver->outpath.c_str(), "ESYS");
                        fprintf(f, "def output_path(x):\n\treturn output_prefix + x\n");
                        fprintf(f, "%s\n", node.child_value());
                        fprintf(f, "%s.run()\n", sim.c_str());
                        fflush(f);
                        fclose(f);
                }

        }
        MPI_Barrier(MPMD.local);

        char * args[] = {fn,NULL};
        
        if (! inter_) {
                output("Spawning esysparticle with script: %s\n", fn);
                inter_ = MPMD.Spawn("esysparticle", args, N, MPI_INFO_NULL);
        }
        acRemoteForceInterface::ConnectRemoteForceInterface("ESYSPARTICLE");
	return 0;
}


// Register the handler (basing on xmlname) in the Handler Factory
template class HandlerFactory::Register< GenericAsk< acESYSParticle > >;
