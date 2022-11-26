#include "acGeometry.h"
std::string acGeometry::xmlname = "Geometry";
#include "../HandlerFactory.h"

int acGeometry::Init () {
		double px=0.0, py=0.0, pz=0.0;
		bool write_pos = false;
		pugi::xml_attribute attr;
		attr = node.attribute("px");
		if (attr) { px = solver->units.alt(attr.value()); write_pos = true; }
		attr = node.attribute("py");
		if (attr) { py = solver->units.alt(attr.value()); write_pos = true; }
		attr = node.attribute("pz");
		if (attr) { pz = solver->units.alt(attr.value()); write_pos = true; }
		if (write_pos) solver->lattice->setPosition(px,py,pz);
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
