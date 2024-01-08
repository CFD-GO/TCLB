#include "Solver.h"

using namespace std;

/// Add a unit gauge
/**
	Add an equation for the units like "1m/s=10"
	\param nm Name of the setting/equation (only for keeping score)
	\param val Value of the setting (eg. 1m/s)
	\param gauge How much it should be equal (eg. 10)
*/
	void Solver::setUnit(std::string nm, std::string val, std::string gauge) {
		units.setUnit(nm, units(val)/units(gauge), 1);
	}

/// Calculate the scales
/**
	Runs the Gauge procedure on the Units object connected to the solver
	This causes the calculate all the unit scales
*/
	void Solver::Gauge() {
		units.makeGauge();
		if (mpi_rank == 0) {
			units.printGauge();
		}
	}

	void Solver::setOutput(std::string out) {
	    auto conffile_stem = std::filesystem::path(conffile_path).stem().string();
	    outpath = out + conffile_stem;
		if (lattice)
		    lattice->snapFileName = std::move(out) + std::move(conffile_stem) +"_Snap";
		NOTICE("Setting output path to: %s\n", outpath.c_str());
	}

/// Inits the csv Log file
/*
	Inits the csv Log file with the header
	/param filename Path to the Log file
*/
	int Solver::initLog(const char * filename)
	{
		if (mpi_rank == 0) {
            FILE * f = NULL;
            debug2("Initializing %s\n",filename);
            f = fopen(filename, "wt");
            assert( f != NULL );
            fprintf(f,"\"Iteration\",\"Time_si\",\"Walltime\",\"Optimization\"");
            LogScales.clear();
            for (auto v : lattice->model->settings) {
                fprintf(f,",\"%s\",\"%s_si\"", v.name.c_str(), v.name.c_str());
                LogScales.push_back(1/units.alt(v.unit));
            }
		    for (auto v : lattice->model->zonesettings) {
		    	for (const auto& [name, id] : setting_zones) {
                    fprintf(f,",\"%s-%s\",\"%s-%s_si\"", v.name.c_str(), name.c_str(), v.name.c_str(), name.c_str());
                }
                LogScales.push_back(1/units.alt(v.unit));
            }
            for (auto v : lattice->model->globals) {
                fprintf(f,",\"%s\",\"%s_si\"", v.name.c_str(), v.name.c_str());
                LogScales.push_back(1/units.alt(v.unit));
            }
            for (auto v : lattice->model->scales) {
                fprintf(f,",\"%s_si\"", v.name.c_str());
                LogScales.push_back(1/units.alt(v.unit));
            }
            fprintf(f,"\n");
            fclose(f);
		}
        return 0;
	}

/// Writes to the csv Log file.
/**
	\param filename Path to the Log file
*/
int Solver::writeLog(const char * filename)
{
    FILE * f = NULL;
    if (mpi_rank == 0) {
        f = fopen(filename, "at");
        assert( f != NULL );
        int j = lattice->model->settings.size() + lattice->model->zonesettings.size() + lattice->model->globals.size() + lattice->model->settings.by_name("dt").id;
        fprintf(f,"%d, %.13le, %.13le, %d",iter, LogScales[j] * iter, get_walltime(), opt_iter);
        j = 0;
        for (auto v : lattice->model->settings) {
            const double val = lattice->getSetting(v.id);
            fprintf(f,", %.13le, %.13le",val,val*LogScales[j]);
            j++;
        }
        for (auto v : lattice->model->zonesettings) {
            for (const auto& [name, id] : setting_zones) {
                int ind = lattice->ZoneIter;
                int zone = id;
                const double val = lattice->zSet.get(v.id, zone, ind);
                fprintf(f,", %.13le, %.13le",val,val*LogScales[j]);
            }
            j++;
        }
        for (auto v : lattice->model->globals) {
            const double val = lattice->globals[v.id];
            fprintf(f,", %.13le, %.13le",val,val*LogScales[j]);
            j++;
        }
        for (auto v : lattice->model->scales) {
            fprintf(f,", %.13le",LogScales[j]);
            j++;
        }
        fprintf(f,"\n");
        fclose(f);
    }
    return 0;
}

CartLattice* Solver::getCartLattice() const {
    if(!lattice) {
        ERROR("Accessing Cartesian lattice before initialization");
        std::terminate();
    }
    auto retval = dynamic_cast<CartLattice*>(lattice.get());
    if(!retval) {
        ERROR("Attempting to access a non-Cartesian Lattice via the Cartesian interface (likely in a handler)");
        std::terminate();
    }
    return retval;
}

ArbLattice* Solver::getArbLattice() const {
    if(!lattice) {
        ERROR("Accessing arbitrary lattice before initialization");
        std::terminate();
    }
    auto retval = dynamic_cast<ArbLattice*>(lattice.get());
    if(!retval) {
        ERROR("Attempting to access a non-arbitrary Lattice via the Arbitrary interface (likely in a handler)");
        std::terminate();
    }
    return retval;
}

/// Generate the connectivity information
/*
	Generate the connectivity information for a 3D torus MPI topology
	\param connect MPI connectivity table to fill
	\param nx Number of segments in X direction
	\param ny Number of segments in Y direction
	\param nz Number of segments in Z direction
*/
static void fillSides(CartConnectivity& connect, int nx, int ny, int nz)
{
	for (int x = 0; x < nx; x++)
	    for (int y = 0; y < ny; y++)
	        for (int z = 0; z < nz; z++) {
		        const int k = x + y * nx + z * nx * ny;
                int j = 0;
                for (int dz = -1; dz <= 1; dz++)
                    for (int dy = -1; dy <= 1; dy++)
                        for (int dx = -1; dx <= 1; dx++){
                            connect.nodes[k].side[j] = ((nx + x - dx) % nx) + ((ny + y - dy) % ny) * nx + ((nz + z - dz) % nz) * nx * ny;
                            j++;
                        }
                assert(j == 27);
            }
}

///	Decompose the lattice for parallel processing
/**
        Divides the lattice into similar-sized parts for MPI parallel processing
*/
static CartConnectivity partitionCartLattice(int nx, int ny, int nz, int n_parts) {
    nx += ThreadsPerBlock::xsdim - 1 - ((nx - 1) % ThreadsPerBlock::xsdim); // nx should be divisible by xsdim
    const float optcom = 2 * sqrt((float)ny * nz * n_parts);
    float mincom = (1 + n_parts) * (ny + nz);
    CartConnectivity retval;
    retval.nodes.resize(n_parts);
    retval.global_region.nx = nx;
    retval.global_region.ny = ny;
    retval.global_region.nz = nz;
    for (int divz = 1; divz <= n_parts; divz++)
        if (n_parts % divz == 0) {
            const int divy = n_parts / divz;
            const float com = divz * ny + divy * nz;
            std::string log = formatAsString("MPI division %d x %d. Communication: %f (%3.0f%%) ", divz, divy, com, 100 * (com / optcom - 1));
            if (com < mincom) {
                mincom = com;
                auto zlens = std::make_unique<int[]>(divz), ylens = std::make_unique<int[]>(divy);
                int mz, my;
                mz = nz;
                my = ny;
                if (mz >= divz && my >= divy) {
                    log += formatAsString("Division:");
                    for (int i = 0; i < divy; i++) {
                        ylens[i] = my / (divy - i);
                        my -= ylens[i];
                        log += formatAsString(" %d", ylens[i]);
                    }
                    log += formatAsString(" x");
                    for (int i = 0; i < divz; i++) {
                        zlens[i] = mz / (divz - i);
                        mz -= zlens[i];
                        log += formatAsString(" %d", zlens[i]);
                    }
                    log += formatAsString("\n");
                    int dz = 0, dy = 0, k = 0;
                    for (int i = 0; i < divz; i++) {
                        dy = 0;
                        for (int j = 0; j < divy; j++) {
                            retval.nodes[k].region.dz = dz;
                            retval.nodes[k].region.dy = dy;
                            retval.nodes[k].region.nz = zlens[i];
                            retval.nodes[k].region.ny = ylens[j];
                            // retval.nodes[k].region.dx = 0; TODO: make sure the default value ok
                            retval.nodes[k].region.nx = nx;
                            dy += ylens[j];
                            k++;
                        }
                        dz += zlens[i];
                    }
                    retval.divx = 1;
                    retval.divy = divy;
                    retval.divz = divz;
                    fillSides(retval, 1, divy, divz);
                } else {
                    log += formatAsString("Mesh too small to divide\n");
                }
            } else {
                log += formatAsString("\n");
            }
                debug2(log.c_str());
        }
    int k = 0;
    for (int i = 0; i < n_parts; i++) {
        debug2("Processor %d will get: %dx%dx%d\n", i, retval.nodes[i].region.nx, retval.nodes[i].region.ny, retval.nodes[i].region.nz);
        if (k < retval.nodes[i].region.size()) k = retval.nodes[i].region.size();
    }
    float overhead = ((double)(k * n_parts - retval.global_region.size())) / retval.global_region.size();
    notice("Max region size: %d. Mesh size %d. Overhead: %2.f%%\n", k, retval.global_region.size(), overhead * 100);

    return retval;
}

static void broadcastCartConnectivity(CartConnectivity& connect, MPI_Comm comm) {
    MPI_Bcast(connect.nodes.data(), connect.nodes.size() * sizeof(NodeInfo), MPI_BYTE, 0, comm);
    MPI_Bcast(&connect.global_region, sizeof(lbRegion), MPI_BYTE, 0, comm);
    MPI_Bcast(&connect.divx, 1, MPI_INT, 0, comm);
    MPI_Bcast(&connect.divy, 1, MPI_INT, 0, comm);
    MPI_Bcast(&connect.divz, 1, MPI_INT, 0, comm);
}

static CartConnectivity makeConnectivity(int nx, int ny, int nz, MPI_Comm comm) {
    int my_rank, comm_sz;
    MPI_Comm_rank(comm, &my_rank);
    MPI_Comm_size(comm, &comm_sz);
    CartConnectivity retval;
    if(my_rank == 0)
        retval = partitionCartLattice(nx, ny, nz, comm_sz);
    else
        retval.nodes.resize(comm_sz);
    broadcastCartConnectivity(retval, comm);
    retval.mpi_rank = my_rank;
    return retval;
}

///	Sets the size of the Lattice
/**
	Sets the size and allocates all the needed buffers etc.
	\param nx X size of the lattice
	\param ny Y size of the lattice
	\param nz Z size of the lattice (1 for 3D)
	\param ns Number of Snapshots allocated
*/
int Solver::initCartLattice(int nx, int ny, int nz) {
    output("Global lattice size: %dx%dx%d\n", nx, ny, nz);
    auto connect = makeConnectivity(nx, ny, nz, mpi_comm);

    const auto& local_region = connect.getLocalRegion();
    output("Local lattice size: %dx%dx%d\n", local_region.nx, local_region.ny, local_region.nz);

    debug0("Creating Lattice object ...");
    lattice = std::make_unique<Lattice<CartLattice>>(std::move(connect), num_snaps, units);
    debug0("Lattice done");

    return 0;
}

/// Initialize arbitrary lattice based on the passed xml node
/**
    /param arb_node ArbitraryLattice xml node
*/
int Solver::initArbLattice(pugi::xml_node arb_node) {
    try {
        debug0("Creating ArbLattice object ...");
        lattice = std::make_unique<Lattice<ArbLattice>>(num_snaps, units, setting_zones, arb_node, mpi_comm);
        debug0("ArbLattice done");
        const auto sz_msg = "Global lattice size: " + std::to_string(lattice->getGlobalSize());
        output(sz_msg.c_str());
        return EXIT_SUCCESS;
    } catch(const std::exception& e) {
        ERROR(e.what());
        return EXIT_FAILURE;
    }
}

Solver::LatticeVariant Solver::getLatticeVariant() const {
    if(!lattice.get()) {
        ERROR("Lattice was requested before it was initialized");
        std::terminate();
    } else if (const auto cart_ptr = dynamic_cast<Lattice<CartLattice>*>(lattice.get()); cart_ptr)
        return {cart_ptr};
    else if (const auto arb_ptr = dynamic_cast<Lattice<ArbLattice>*>(lattice.get()); arb_ptr)
        return {arb_ptr};
    else {
        ERROR("Solver owns unknown lattice type, this should not happen");
        std::terminate();
    }
}
