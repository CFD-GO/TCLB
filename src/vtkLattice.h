#ifndef VTKLATTICE_H

	#include "Global.h"
	#include "Node.h"
	#include "LatticeContainer.h"
	#include "Lattice.h"
	#include "vtkOutput.h"
	#include "unit.h"

	int vtkWriteLattice(char * filename, Lattice * lattice, UnitEnv);
	void screenDumpLattice(Lattice * lattice);
	int initMean(char * filename);
	int writeMean(char * filename, Lattice * lattice, int, int iter, double);

#endif
#define VTKLATTICE_H 1
