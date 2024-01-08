#include "acGeometry.h"
std::string acGeometry::xmlname = "Geometry";
#include "../HandlerFactory.h"

int acGeometry::Init () {
		double px=0.0, py=0.0, pz=0.0;
		bool write_pos = false;
                const auto lattice = solver->getCartLattice();
		pugi::xml_attribute attr;
		attr = node.attribute("px");
		if (attr) { px = solver->units.alt(attr.value()); write_pos = true; }
		attr = node.attribute("py");
		if (attr) { py = solver->units.alt(attr.value()); write_pos = true; }
		attr = node.attribute("pz");
		if (attr) { pz = solver->units.alt(attr.value()); write_pos = true; }
		if (write_pos) lattice->setPosition(px,py,pz);
			if (lattice->geometry->load(node, solver->setting_zones)) {
				error("Error while loading geometry\n");
				return -1;
			}
			lattice->setFlags(lattice->geometry->region,lattice->geometry->geom);
			lattice->CutsOverwrite(lattice->geometry->Q,lattice->geometry->region);
			lattice->zSet.zone_max(solver->setting_zones.size() - 1);
			return 0;
	}


// Register the handler (basing on xmlname) in the Handler Factory
template class HandlerFactory::Register< GenericAsk< acGeometry > >;
