#ifndef SOLVER_H
#define SOLVER_H

#include "Consts.h"
#include "pugixml.hpp"
#include "Global.h"
#include <mpi.h>
#include "cross.h"
#include "CartLattice.h"
#include "ArbLattice.hpp"
#include "utils.h"
#include "unit.h"
#include "Handlers.h"
#include "Lattice.hpp"

#include <cassert>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <variant>
#include <vector>

/// Main solver class
/**
 This class is responsible for the solver initialization and run
*/
class Solver {
public:
    std::unique_ptr<LatticeBase> lattice; ///< Main Lattice object
    MPI_Comm mpi_comm; ///< Solver's MPMD.local
    int iter = 0; ///< Iteration (Now)
    int opt_iter = 0; ///< Optimization iteration
	int mpi_rank = 0; ///< MPI rank
	int mpi_size = 1; ///< MPI size
	std::vector<Handler> hands; ///< Stack of handlers
	int steps = 1; ///< steps to the next monitor-callback
	int saveN = 0, saveI = 0; ///< No idea what it is TODO
	int num_snaps = 2; ///< Number of snaps, this is passed down to the lattice member
	pugi::xml_document configfile;
	std::string conffile_path; ///< Path to the config file
    std::string outpath; ///< Output prefix
	UnitEnv units; ///< Units object connected to this lattice
	std::map<std::string, int> setting_zones = {{"DefaultZone", 0}};
	int iter_type = ITER_NORM; ///< Iteration type (Now) - primal/adjoint/etc.
    std::vector<double> LogScales;

    Solver() = default;
    Solver(const Solver&) = delete;
    Solver(Solver&&) = default;
    Solver& operator=(const Solver&) = delete;
    Solver& operator=(Solver&&) = default;
    ~Solver() = default;

    int initCartLattice(int nx, int ny, int nz);
    int initArbLattice(pugi::xml_node arb_node);
    CartLattice* getCartLattice() const;
    ArbLattice* getArbLattice() const;

    using LatticeVariant = std::variant<Lattice<CartLattice>*, Lattice<ArbLattice>*>;
    LatticeVariant getLatticeVariant() const;

    int EventLoop() const { return lattice->EventLoop(); }
	void print(const char * str);

	size_t getLocalLatticeSize() const { return lattice->getLocalSize(); }
	size_t getGlobalLatticeSize() const { return lattice->getGlobalSize(); }

    void initMPI(MPI_Comm comm);
    int initLog(const char * filename);
    int writeLog(const char * filename);

/// Generate a Iteration-specific filename
/**
 Generate a filename starting with the output prefix, continuing with the name, process number,
 iteration number and suffix
 \param name Appendix added to the filename
 \param suffix Suffix (.vti, .csv, etc) of the file
 \param out Buffer for the returned file name
*/
    std::string outIterFile(const std::string& name, const std::string& suffix) const {
        auto path = formatAsString("%s_%s_P%02d_%08d%s", outpath, name, mpi_rank, iter, suffix);
		mkdir_p(path);
		return path;
    }

/// Generate a Case-specific filename
/**
 Generate a filename starting with the output prefix, continuing with the name, process number,
 and finishing with a suffix
 \param name Appendix added to the filename
 \param suffix Suffix (.vti, .csv, etc) of the file
 \param out Buffer for the returned file name
*/
    std::string outGlobalFile(const std::string& name, const std::string& suffix) const {
        auto path = formatAsString("%s_%s_P%02d%s", outpath, name, mpi_rank, suffix);
        mkdir_p(path);
        return path;
    }

/// Generate a Iteration-specific collective filename
/**
 Generate a filename starting with the output prefix, continuing with the name,
 iteration number and suffix (without process number)
 \param name Appendix added to the filename
 \param suffix Suffix (.vti, .csv, etc) of the file
 \param out Buffer for the returned file name
*/
    std::string outIterCollectiveFile(const std::string& name, const std::string& suffix) {
        auto path = formatAsString("%s_%s_%08d%s", outpath, name, iter, suffix);
        mkdir_p(path);
        return path;
    }

	void setOutput(std::string out);
	void setUnit(std::string, std::string, std::string);
	void Gauge();
};

#endif
