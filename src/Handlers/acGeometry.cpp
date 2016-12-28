#include "acGeometry.h"
std::string acGeometry::xmlname = "Geometry";
#include "../HandlerFactory.h"

int acGeometry::Init () {
			if (solver->geometry->load(node)) {
				error("Error while loading geometry\n");
				return -1;
			}
			solver->lattice->FlagOverwrite(solver->geometry->geom,solver->geometry->region);
			solver->lattice->CutsOverwrite(solver->geometry->Q,solver->geometry->region);
			solver->lattice->zSet.zone_max(solver->geometry->SettingZones.size()-1);
			return 0;
	}


// Register the handler (basing on xmlname) in the Handler Factory
template class HandlerFactory::Register< GenericAsk< acGeometry > >;
