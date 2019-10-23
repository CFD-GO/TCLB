#ifndef HDF5LATTICE_H

	#include "Global.h"
//	#include "LatticeContainer.h"
	#include "Solver.h"
	#include "unit.h"

	int hdf5WriteLattice(const char * filename, Solver * solver, name_set * s, bool write_xdmf);

#endif
#define HDF5LATTICE_H 1
