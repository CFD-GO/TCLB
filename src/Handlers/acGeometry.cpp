#include "acGeometry.h"
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


// Function created only to check to create Handler for specific conditions
vHandler * Ask_For_acGeometry(const pugi::xml_node& node) {
  std::string name = node.name();
  if (name == "Geometry") {
		return new acGeometry;
  }
  return NULL;
}

// Register this function in the Handler Factory
template class HandlerFactory::Register< Ask_For_acGeometry >;

