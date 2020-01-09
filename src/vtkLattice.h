#ifndef VTKLATTICE_H

	#include "Global.h"
//	#include "LatticeContainer.h"
	#include "Lattice.h"
	#include "vtkOutput.h"
	#include "unit.h"
	#include "utils.h"

	int vtkWriteLattice(char * filename, LatticeBase * lattice, UnitEnv, name_set * s);
	int binWriteLattice(char * filename, LatticeBase * lattice, UnitEnv units);
	int txtWriteLattice(char * filename, LatticeBase * lattice, UnitEnv, name_set * s, int type);
	void screenDumpLattice(LatticeBase * lattice);
	int initMean(char * filename);
	int writeMean(char * filename, LatticeBase * lattice, int, int iter, double);

#endif
#define VTKLATTICE_H 1
