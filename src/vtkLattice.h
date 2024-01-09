#ifndef VTKLATTICE_H
#define VTKLATTICE_H

#include <string_view>

#include "ArbLattice.hpp"
#include "CartLattice.h"
#include "Global.h"
#include "unit.h"
#include "utils.h"
#include "vtkOutput.h"
#include "vtuOutput.h"

int vtkWriteLattice(const std::string& filename, CartLattice& lattice, const UnitEnv&, const name_set& s, const lbRegion& region);
int vtuWriteLattice(const std::string& filename, ArbLattice& lattice, const UnitEnv&, const name_set& s);
int binWriteLattice(const std::string& filename, LatticeBase& lattice, const UnitEnv& units);
int txtWriteLattice(const std::string& filename, LatticeBase& lattice, const UnitEnv&, const name_set& s, int type);
void screenDumpLattice(const CartLattice& lattice);
int initMean(const std::string& filename);
int writeMean(const std::string& filename, const CartLattice& lattice, int, int iter, double);

#endif
