#ifndef HDF5LATTICE_H

	#include "Global.h"
//	#include "CartLatticeContainer.h"
	#include "Solver.h"
	#include "unit.h"

	#define HDF5_DEFLATE 0x01
	#define HDF5_WRITE_XDMF 0x02
	#define HDF5_WRITE_DOUBLE 0x04
	#define HDF5_WRITE_LBM 0x08
	#define HDF5_WRITE_POINT 0x10
	
	int hdf5WriteLattice(const char * filename, Solver * solver, name_set * s, unsigned long int* chunkdim_, unsigned int options, lbRegion region);

#endif
#define HDF5LATTICE_H 1
