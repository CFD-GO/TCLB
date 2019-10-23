#ifndef HDF5LATTICE_H

	#include "Global.h"
//	#include "LatticeContainer.h"
	#include "Lattice.h"
	#include "unit.h"

	int hdf5WriteLattice(const char * filename, Lattice * lattice, UnitEnv, name_set * s);

#endif
#define HDF5LATTICE_H 1
