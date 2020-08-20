#include "acConnectivity.h"
std::string acConnectivity::xmlname = "ArbitraryLattice";
#include "../HandlerFactory.h"

int acConnectivity::Init () {
			solver->connectivity->load(node);
			solver->lattice->LoadLattice(solver->connectivity->connectivity, solver->connectivity->coords, solver->connectivity->geom, 
											solver->connectivity->latticeSize, solver->connectivity->Q);
			return 0;
	}


// Register the handler (basing on xmlname) in the Handler Factory
template class HandlerFactory::Register< GenericAsk< acConnectivity > >;
