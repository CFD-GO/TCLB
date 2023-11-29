#ifndef VTKLATTICE_H

#include "Global.h"
#include "CartLattice.h"
#include "vtkOutput.h"
#include "unit.h"
#include "utils.h"

#include <string_view>

int vtkWriteLattice(const std::string& filename, CartLattice& lattice, const UnitEnv&, const name_set& s, const lbRegion& region);
int binWriteLattice(const std::string& filename, CartLattice& lattice, const UnitEnv& units);
int txtWriteLattice(const std::string& filename, CartLattice& lattice, const UnitEnv&, const name_set& s, int type);
void screenDumpLattice(const CartLattice& lattice);
int initMean(const std::string& filename);
int writeMean(const std::string& filename, const CartLattice& lattice, int, int iter, double);

#endif
#define VTKLATTICE_H 1
