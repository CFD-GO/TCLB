#ifndef VTKLATTICE_H

	#include "Global.h"
	#include "LatticeBase.h"
	#include "Connectivity.h"
	#include "vtkOutput.h"
	#include "vtpOutput.h"
	#include "vtuOutput.h"
	#include "unit.h"
	#include "utils.h"

	int vtkWriteLattice(char * filename, LatticeBase * lattice, UnitEnv, name_set * s);
	int vtkWriteLatticeArbitrary(char * filename, size_t latticeSize, LatticeBase * lattice, UnitEnv, name_set * s);
	int vtkWriteLatticeArbitraryUG(char * filename, size_t latticeSize, LatticeBase * lattice, Connectivity * connectivity, UnitEnv, name_set * s);
	int binWriteLattice(char * filename, LatticeBase * lattice, UnitEnv units);
	int txtWriteLattice(char * filename, LatticeBase * lattice, UnitEnv, name_set * s, int type);
	
	//void WriteStructuredGridField(FILE* f, size_t size, const char* name, void* data, int elem, const char* tp, int components);
	//void WriteSGB64(FILE* f, void* data, int len);
	//void fprintSGB64(FILE* f, void * tab, int len);
	//void Base64char3_arb(unsigned char * in, int len, char * out);

	void screenDumpLattice(LatticeBase * lattice);
	int initMean(char * filename);
	int writeMean(char * filename, LatticeBase * lattice, int, int iter, double);

#endif
#define VTKLATTICE_H 1
