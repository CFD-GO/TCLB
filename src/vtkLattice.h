#ifndef VTKLATTICE_H

	#include "Global.h"
//	#include "LatticeContainer.h"
	#include "Lattice.h"
	#include "vtkOutput.h"
	#include "unit.h"
	#include "utils.h"

	int vtkWriteLattice(char * filename, Lattice * lattice, UnitEnv, name_set * s);
	int binWriteLattice(char * filename, Lattice * lattice, UnitEnv units);
	int txtWriteLattice(char * filename, Lattice * lattice, UnitEnv, name_set * s, int type);
	void screenDumpLattice(Lattice * lattice);
	int initMean(char * filename);
	int writeMean(char * filename, Lattice * lattice, int, int iter, double);

#endif
#define VTKLATTICE_H 1
