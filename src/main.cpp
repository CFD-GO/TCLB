/*  Main program file                                          */
/*     Here we have all the initialization and the main loop   */
/*-------------------------------------------------------------*/

#include "Consts.h"

#ifdef EMBEDED_PYTHON
#include <Python.h>
#endif

#include "Global.h"
#include <mpi.h>
#include "Region.h"
#include "utils.h"
#include "unit.h"

#include <fstream>
#include <iostream>
#include <vector>
#include <iomanip>
#include <assert.h>

#include <ctime>
#include "glue.hpp"
// for isatty
//#include <unistd.h>

#include "Solver.h"
#include "xpath_modification.h"
#include "mpitools.hpp"

// Reads units from configure file and applies them to the solver
int readUnits(pugi::xml_node config, Solver* solver) {
	pugi::xml_node set = config.child("Units");
	if (!set) {
		warning("No \"Units\" element in config file\n");
		return 0;
	}
	int i=1;
	for (pugi::xml_node node = set.child("Param"); node; node = node.next_sibling("Param")) {
	        std::string par, value, gauge;
		pugi::xml_attribute attr;
		attr = node.attribute("value");
		if (attr) {
			value = attr.value();
		} else {
			ERROR("Value not provided in Param in Units\n");
			return -1;
		}
		attr = node.attribute("gauge");
		if (attr) {
			gauge = attr.value();
		} else {
			ERROR("Gauge not provided in Param in Units\n");
			return -1;
		}
		attr = node.attribute("name");
		if (attr) par = attr.value(); else par = (Glue() << "unnamed" << i).str();
		debug2("Units: %s = %s = %s\n", par.c_str(), value.c_str(), gauge.c_str());
		solver->setUnit(par, value, gauge);
		i++;
	}
	solver->Gauge();
	return 0;
};

CudaEvent_t     start, stop; // CUDA events to measure time

// Main callback function called every some iterations to display speed
int MainCallback(int seg, int tot, Solver* solver) {
	int begin=false, end=false;

	if (tot ==   0) begin=true;

	if (tot == seg) end=true;
	
	static int iter=0;
	static int last_tot=0;
	if (tot<last_tot) last_tot = 0;
	iter += tot-last_tot;
	last_tot = tot;
	static double cum_time = 0;
	static int steps = 1;
	if ((iter < steps) & (!end)) return -1;
	if (iter == 0) return -2;
	float   elapsedTime=0; // Elapsed times
	CudaEventRecord( stop, 0 );
	CudaEventSynchronize( stop );
	CudaEventElapsedTime( &elapsedTime, start, stop );
	
	if (D_MPI_RANK == 0) {
		int desired_steps;
		float eTime;
		int all_iter;
		cum_time += elapsedTime;
		if (end) {
			eTime = cum_time;
			all_iter = tot;
		} else {
			eTime = elapsedTime;
			all_iter = iter;
		}
       		int ups = (float) (1000. * all_iter)/eTime; // Steps made per second
  		double lbups=1.0;
       		lbups *= solver->info.region.nx;
		lbups *= solver->info.region.ny;
       		lbups *= solver->info.region.nz;
       		lbups *= iter;
		lbups /= elapsedTime;
		desired_steps = ups/desired_fps; // Desired steps per frame (so that on next frame fps = desired_fps)
		char per[1000];
		char buf[1000];
		char left[1000];
		//int left_s = (cum_time * (seg - tot)) / ((tot+1) * 1000);
		int left_s = cum_time/1000;
		if (left_s < 60) {
				sprintf(left,      "%2ds",         left_s);
		} else {
			int left_m = left_s/60;
			left_s = left_s - left_m*60;
			if (left_m < 60) {
				sprintf(left, "%2dm %2ds", left_m, left_s);
			} else {
				int left_h = left_m/60;
				left_m = left_m - left_h*60;
				sprintf(left,  "%dh %2dm", left_h, left_m);
			}
		}
		sprintf(buf, "%8.1f MLBUps   %7.2f GB/s", ((double)lbups)/1000, ( (double) lbups * ((double) 2 * NUMBER_OF_DENSITIES * sizeof(real_t) + sizeof(flag_t))) / 1e6);
		int per_len = 20;
		{
			int i=0;
			per[i]='['; i++;
			for (; i <= per_len; i++) if (i * seg <= tot * per_len) per[i]='='; else per[i]=' ';
			per[i]=']'; i++;
			per[i]=0;
		}
		if (end) {
			output("%s %s\n", buf, per);
			cum_time = 0;
		} else {
			if (D_TERMINAL) {
				printf("[  ] %s %s %s\r", buf, per, left);
			} else {
				printf("[  ] %s %s %s\n", buf, per, left);
			}
		}
		fflush(stdout);
		steps = desired_steps;
		if (steps < 1) steps = 1;
		if (steps % 2 == 1) steps ++;
	}
	MPI_Bcast(&steps, 1, MPI_INT, 0, MPMD.local);
	solver->EventLoop();
	CudaEventRecord( start, 0 );
	CudaEventSynchronize( start );
	iter = 0;
	return steps;
}

// Finds the adjoint element in config to know how many snaps to allocate
bool find_adjoint(pugi::xml_node node)
{
	if (strcmp(node.name(), "Adjoint") == 0) {
		if (strcmp(node.attribute("type").value(), "steady") == 0) {
			return false;
		} else {
			return true;
		}
	} else return false;
}


// Main program function
int main ( int argc, char * argv[] )
{
	Solver * solver;

	// Error handling for scanf
	#define HANDLE_IOERR(x) if ((x) == EOF) { error("Error in fscanf.\n"); return -1; }
	MPI_Init(&argc, &argv);
	MPMD.Init(MPI_COMM_WORLD,"TCLB");
	MPMD.Identify();

	solver = new Solver(MPMD.local); // Global data declaration

	MPI_Comm_rank(MPMD.local,  &solver->mpi_rank);
	MPI_Comm_size(MPMD.local,  &solver->mpi_size);
	DEBUG_SETRANK(solver->mpi_rank);
	DEBUG_M;
	InitPrint(DEBUG_LEVEL, 6, 8);
	MPI_Barrier(MPMD.local);

	global_start = std::clock();
	if (solver->mpi_rank == 0) {
		NOTICE("-------------------------------------------------------------------------\n");
		NOTICE("-  CLB version: %25s                               -\n",VERSION);
		NOTICE("-        Model: %25s                               -\n",MODEL);
		NOTICE("-------------------------------------------------------------------------\n");
	}
	MPI_Barrier(MPMD.local);
	DEBUG_M;

	DEBUG0(
	debug0("0 level debug");
	debug1("1 level debug");
	debug2("2 level debug");
	output("normal output");
	notice("notice");
	NOTICE("important notice");
	warning("warning");
	WARNING("important warning");
	error("error");
	ERROR("fatal error");
	)
	//Prepare MPI solver-structure
	solver->mpi.node = new NodeInfo[solver->mpi_size];
	solver->mpi.size = solver->mpi_size;
	solver->mpi.rank = solver->mpi_rank;
	solver->mpi.gpu = 0;
	for (int i=0;i < solver->mpi_size; i++) solver->mpi.node[i].rank = i;

	// Reading arguments
	// At least one argument
	if ( argc < 2 ) {
		error("Not enough parameters");
		notice("Usage: program configfile [device number]\n");
		return 0;
	}

	// Calculating the right number of threads per block
	#ifdef CROSS_CPU
		solver->info.xsdim = 1;
		solver->info.ysdim = 1;
	#else
		solver->info.xsdim = 32;
		solver->info.ysdim = 1;
	#endif

	debug0("sizeof(size_t) = %ld\n", sizeof(size_t));
	debug0("sizeof(real_t) = %ld\n", sizeof(real_t));
	debug0("sizeof(vector_t) = %ld\n", sizeof(vector_t));
	debug0("sizeof(flag_t) = %ld\n", sizeof(flag_t));

	MPI_Barrier(MPMD.local);

	// Reading the config file
	char* filename = argv[1];
//	pugi::xml_document configfile;
        if (xml_def_init()) { error("Error in xml_def_init. It should work!\n"); return -1; }
	strcpy(solver->info.conffile, filename);
	solver->setOutput("");
	pugi::xml_parse_result result = solver->configfile.load_file(filename, pugi::parse_default | pugi::parse_comments);
	if (!result) {
		error("Error while parsing %s: %s\n", filename, result.description());
		return -1;
	}
	
	#define XMLCHILD(x,y,z) { x = y.child(z); if (!x) { error(" in %s: No \"%s\" element\n",filename,z); return -1; }}
	pugi::xml_node config, geom, units;
	XMLCHILD(config, solver->configfile, "CLBConfig");
	
	// Treatment of depreciated Params element:
	{
		pugi::xpath_node_set found = config.select_nodes("//Params");
		if (found.size() > 0) {
                    	WARNING("%ld depreciated Params elements found. Changing them to Param:\n", found.size());
                    	config.append_attribute("permissive").set_value("true");
                    	for (pugi::xpath_node_set::const_iterator it = found.begin(); it != found.end(); ++it) {
				pugi::xml_node node = it->node();
				pugi::xml_node after = node;
				std::string gauge = "";
				pugi::xml_attribute gauge_attr = node.attribute("gauge");
				if (gauge_attr) gauge = gauge_attr.value();
				for (pugi::xml_attribute attr = node.first_attribute(); attr; attr = attr.next_attribute()) if (strcmp(attr.name(),"gauge") != 0) {
				        std::string par, zone;
			                par = attr.name();
		                        size_t i = par.find_first_of('-');
		                        if (i == string::npos) {
		                        	zone = "";
		                        } else {
                		                zone = par.substr(i+1);
		                                par = par.substr(0,i);
					}
					pugi::xml_node param = node.parent().insert_child_after("Param", after);
					after = param;
					param.append_attribute("name").set_value(par.c_str());
					param.append_attribute("value").set_value(attr.value());
					if (zone != "") param.append_attribute("zone").set_value(zone.c_str());
					if (gauge != "") param.append_attribute("gauge").set_value(gauge.c_str());
				}
				node.parent().remove_child(node);
			}
		}
	}	


	if (argc > 2) {
		int status = xpath_modify(solver->configfile, config, argc-2, argv+2);
		if (status == -444) { // Graceful exit
			if (readUnits(config, solver)) {
				ERROR("Wrong Units\n");
				return -1;
			}
			return 0;
		}
		if (status != 0) return status;
	}

	// Delete comments:
	{
		pugi::xpath_node_set found = config.select_nodes("//comment()");
		if (found.size() > 0) {
                    	output("Discarding %ld comments\n", found.size());
                    	for (pugi::xpath_node_set::const_iterator it = found.begin(); it != found.end(); ++it) {
				pugi::xml_node node = it->node();
				if (node) {
					node.parent().remove_child(node);
				} else {
					ERROR("Comment is not a node (this should not happen)\n");
				}
			}
		}
	}

	// After the configfile comes the numbers of GPU selected for each processor (starting with 0)
	{
		#ifndef CROSS_CPU
			int count, dev;
			CudaGetDeviceCount( &count );
			{
				MPI_Comm comm = MPMD.local;
				std::string nodename = mpitools::MPI_Nodename(comm);
				MPI_Comm nodecomm = mpitools::MPI_Split(nodename, comm);
				dev = mpitools::MPI_Rank(nodecomm);
				MPI_Comm_free(&nodecomm);
			}
			if (dev >= count) {
				bool oversubscribe = false;
				pugi::xml_attribute attr = config.attribute("oversubscribe_gpu");
				if (attr) oversubscribe = attr.as_bool();
				if (!oversubscribe) {
					ERROR("Oversubscribing GPUs. This is not a good idea, but if you want to do it, add oversubscribe_gpu=\"true\" to the config file");
					return -1;
				} else {
					WARNING("Oversubscribing GPUs.");
				}
				dev = dev % count;
			}
			output_all("Selecting device %d/%d\n", dev, count);
			CudaSetDevice( dev );
			solver->mpi.gpu = dev;
			debug2("Initializing device\n");
			cudaFree(0);
		#else
			output_all("Running on CPU\n");
			CudaSetDevice(0);
			solver->mpi.gpu = 0;
		#endif
	}
	MPI_Barrier(MPMD.local);
	DEBUG_M;

	
	if (readUnits(config, solver)) {
		ERROR("Wrong Units\n");
		return -1;
	}

	XMLCHILD(geom, config, "Geometry");

	// Reading the size of mesh
	int nx, ny, nz, ns = 2;
	nx = myround(solver->units.alt(geom.attribute("nx").value(),1));
	ny = myround(solver->units.alt(geom.attribute("ny").value(),1));
	nz = myround(solver->units.alt(geom.attribute("nz").value(),1));
	notice("Mesh size in config file: %dx%dx%d\n",nx,ny,nz);

	// Finding the adjoint element
	pugi::xml_node adj;
	adj = solver->configfile.find_node(find_adjoint);
	if (adj) {
		pugi::xml_attribute attr = adj.attribute("NumberOfSnaps");
		if (attr) {
			ns = attr.as_int();
		} else {
			ns = 10;
		}
		if (ns < 2) ns =2;
		NOTICE("Will be running nonstationary adjoint at %d Snaps\n", D_MPI_RANK, ns);
	}

	// Initializing the lattice of a specific size
	if (solver->setSize(nx,ny,nz,ns)) return -1;
	solver->setOutput("");

	//Setting settings to default
	// Initializing the CUDA events and setting callback
	CudaEventCreate( &start );
	CudaEventCreate( &stop );
	CudaEventRecord( start, 0 );
	solver->lattice->Callback((int(*)(int, int, void*)) MainCallback, (void*) solver);

	// Running main handler (it makes all the magic)
	{
		Handler hand(config, solver);
		if (!hand) {
			error("Something went wrong in xml run!\n");
			return -1;
		}
	}
    #ifdef EMBEDED_PYTHON
    Py_Finalize();
    #endif

	// Finish and clean up
	debug2("cudaFree ...\n");
	CudaEventDestroy( start );
	CudaEventDestroy( stop );

	if (solver->mpi_rank == 0) {
		double duration = (std::clock() - global_start) / (double)CLOCKS_PER_SEC;
		output("Total duration: %lf s = %lf min = %lf h\n", duration, duration / 60, duration /60/60);
	}
	delete solver;
	CudaDeviceReset();
	MPI_Finalize();
	return 0;
}


