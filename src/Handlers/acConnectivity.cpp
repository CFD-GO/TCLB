#include "acConnectivity.h"
std::string acConnectivity::xmlname = "ArbitraryLattice";
#include "../HandlerFactory.h"

int acConnectivity::Init () {
			/*if (solver->geometry->load(node)) {
				error("Error while loading geometry\n");
				return -1;
			}
			solver->lattice->FlagOverwrite(solver->geometry->geom,solver->geometry->region);
			solver->lattice->CutsOverwrite(solver->geometry->Q,solver->geometry->region);
			solver->lattice->zSet.zone_max(solver->geometry->SettingZones.size()-1);*/

			solver->connectivity->load(node);

			return 0;
	}


// Register the handler (basing on xmlname) in the Handler Factory
template class HandlerFactory::Register< GenericAsk< acConnectivity > >;
