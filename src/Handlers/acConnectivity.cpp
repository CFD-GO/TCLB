#include "acConnectivity.h"
std::string acConnectivity::xmlname = "ArbitraryLattice";
#include "../HandlerFactory.h"

int acConnectivity::Init () {
			solver->connectivity->load(node);
			solver->lattice->LoadLattice(solver->connectivity->connectivity, solver->connectivity->coords, solver->connectivity->geom, 
											solver->connectivity->connectivityDirections, solver->connectivity->latticeSize, solver->connectivity->Q,
											solver->connectivity->ndx, solver->connectivity->ndy, solver->connectivity->ndz,
											solver->connectivity->mindx, solver->connectivity->mindy, solver->connectivity->mindz);
			return 0;
	}


// Register the handler (basing on xmlname) in the Handler Factory
template class HandlerFactory::Register< GenericAsk< acConnectivity > >;
