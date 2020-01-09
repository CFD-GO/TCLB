#ifndef SOLVER_H
#define SOLVER_H

#include "Consts.h"
#include "pugixml.hpp"
#include "Global.h"
#include <mpi.h>
#include "cross.h"
#include "Region.h"
#include "LatticeContainer.h"
#include "Lattice.h"
#include "vtkLattice.h"
#include "Geometry.h"
#include "def.h"
#include "utils.h"
#include "unit.h"

#include <fstream>
#include <iostream>
#include <vector>
#include <iomanip>
#include <assert.h>
#include "Handlers.h"

#ifdef GRAPHICS
    const int desired_fps = 10;
	class GPUAnimBitmap;
#else
    const int desired_fps = 1;
#endif

using namespace std;

/// Class storing all the processor-common information
/**
 It is used to broadcast all the common information among the processes
 */
struct InfoBlock {
	char conffile[STRING_LEN]; ///< Path to the config file
	lbRegion region; ///< Global region of the lattice
	int xsdim,ysdim; ///< X and Y thread divisions
        char outpath[STRING_LEN]; ///< Output prefix
};


/// Main solver class
/**
 This class is responsible for the solver initialization and run
*/
class Solver {
    public:
	InfoBlock info; ///< Information common among the cores
	MPIInfo mpi; ///< Information on MPI connectivity
	MPI_Comm mpi_comm; ///< Solver's MPMD.local
	pugi::xml_document configfile;
        LatticeBase * lattice; ///< Main Lattice object
	Geometry * geometry; ///< Main Geometry object
	lbRegion region; ///< Global region
        int iter; ///< Iteration (Now)
        int opt_iter; ///< Optimization iteration
	int mpi_rank; ///< MPI rank
	int mpi_size; ///< MPI size
	std::vector<Handler> hands; ///< Stack of handlers
	int steps; ///< steps to the next monitor-callback
	int saveN, saveI; ///< No idea what it is TODO
	char ** saveFile; ///< It shouldn't be here TODO
	UnitEnv units; ///< Units object connected to this lattice
	int iter_type; ///< Iteration type (Now) - primal/adjoint/etc.
#ifdef GRAPHICS
	GPUAnimBitmap * bitmap; ///< Maybe we have a bitmap for animation
#endif
	void print(const char * str);
	double LogScales[ GLOBALS + SETTINGS + ZONESETTINGS + SCALES ];
	
	inline Solver() : mpi_comm(MPMD.local), lattice(NULL) { Init(); };
	~Solver();
	inline Solver(MPI_Comm mpi_comm_) : mpi_comm(mpi_comm_), lattice(NULL) { Init(); };
	void Init();
	void saveInit(int n);
        inline void setWidth(int &w){region.nx = w;};
        inline void setHeight(int &h){region.ny = h;};
        inline int getWidth(){return region.nx;};
        inline int getHeight(){return region.ny;};
/// Generate a Iteration-specific filename
/**
 Generate a filename starting with the output prefix, continuing with the name, process number,
 iteration number and suffix
 \param name Appendix added to the filename
 \param suffix Suffix (.vti, .csv, etc) of the file
 \param out Buffer for the returned file name
*/
        inline void outIterFile(const char * name, const char * suffix, char * out) {
                sprintf(out, "%s_%s_P%02d_%08d%s", info.outpath, name, mpi_rank, iter, suffix);
		mkpath(out);
        };
/// Generate a Case-specific filename
/**
 Generate a filename starting with the output prefix, continuing with the name, process number,
 and finishing with a suffix
 \param name Appendix added to the filename
 \param suffix Suffix (.vti, .csv, etc) of the file
 \param out Buffer for the returned file name
*/
        inline void outGlobalFile(const char * name, const char * suffix, char * out) {
                sprintf(out, "%s_%s_P%02d%s", info.outpath, name, mpi_rank, suffix);
		mkpath(out);
        };
/// Generate a Iteration-specific collective filename
/**
 Generate a filename starting with the output prefix, continuing with the name,
 iteration number and suffix (without process number)
 \param name Appendix added to the filename
 \param suffix Suffix (.vti, .csv, etc) of the file
 \param out Buffer for the returned file name
*/
        inline void outIterCollectiveFile(const char * name, const char * suffix, char * out) {
                sprintf(out, "%s_%s_%08d%s", info.outpath, name, iter, suffix);
		mkpath(out);
        };

/// Set output prefix
	void setOutput(const char * out);
	void setUnit(std::string, std::string, std::string);
	void Gauge();
	int initLog(const char * filename);
	int writeLog(const char * filename);
	int writeVTK(const char * nm, name_set * s);
	int writeTXT(const char * nm, name_set * s, int type);
	int writeBIN(const char * nm);
	int setSize(int,int,int,int);
	int MPIDivision();
	int InitAll(int);
	int RunMainLoop();
	int EventLoop();

};
#endif
