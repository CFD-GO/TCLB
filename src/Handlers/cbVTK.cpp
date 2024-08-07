#include "cbVTK.h"
std::string cbVTK::xmlname = "VTK";
#include "../HandlerFactory.h"

int cbVTK::Init () {
		Callback::Init();
		pugi::xml_attribute attr = node.attribute("name");
		nm = "VTK";
		if (attr) nm = attr.value();
		attr = node.attribute("what");
		if (attr) {
			s.add_from_string(attr.value(),',');
		} else {
			s.add_from_string("all",',');
		}

		reg = solver->mpi.totalregion;
	
		attr = node.attribute("dx");
		if (attr) { reg.dx = solver->units.alt(attr.value()); }
		if (reg.dx < 0) {
			reg.dx = reg.nx + reg.dx;
			reg.nx = reg.nx - reg.dx;
		}
		attr = node.attribute("dy");
		if (attr) { reg.dy = solver->units.alt(attr.value()); }
		if (reg.dy < 0) {
			reg.dy = reg.ny + reg.dy;
			reg.ny = reg.ny - reg.dy;
		}
		attr = node.attribute("dz");
		if (attr) { reg.dz = solver->units.alt(attr.value()); }
		if (reg.dz < 0) {
			reg.dz = reg.nz + reg.dz;
			reg.nz = reg.nz - reg.dz;
		}

		attr = node.attribute("nx");
		if (attr) { reg.nx = solver->units.alt(attr.value()); }
		if (reg.nx < 0) { reg.nx = solver->mpi.totalregion.nx - reg.dx + reg.nx; }
		attr = node.attribute("ny");
		if (attr) { reg.ny = solver->units.alt(attr.value()); }
		if (reg.ny < 0) { reg.ny = solver->mpi.totalregion.ny - reg.dy + reg.nz; }
		attr = node.attribute("nz");
		if (attr) { reg.nz = solver->units.alt(attr.value()); }
		if (reg.nz < 0) { reg.nz = solver->mpi.totalregion.nz - reg.dz + reg.nz; }

		reg = reg.intersect(solver->mpi.totalregion);

		debug1("VTK \"%s\" with output region: %dx%dx%d + %d,%d,%d from total region %dx%dx%d + %d,%d,%d", nm.c_str(), 
		reg.nx,reg.ny,reg.nz,reg.dx,reg.dy,reg.dz,solver->mpi.totalregion.nx,solver->mpi.totalregion.ny,solver->mpi.totalregion.nz,solver->mpi.totalregion.dx,solver->mpi.totalregion.dy,solver->mpi.totalregion.dz);
		if (reg.size() == 0) {
			ERROR("VTK \"%s\" output has size 0", nm.c_str());
			return -1;
		}

		return 0;
	}


int cbVTK::DoIt () {
		Callback::DoIt();
		return solver->writeVTK(nm.c_str(), &s, reg);
	};


// Register the handler (basing on xmlname) in the Handler Factory
template class HandlerFactory::Register< GenericAsk< cbVTK > >;
