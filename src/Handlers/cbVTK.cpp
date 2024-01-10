#include "cbVTK.h"

std::string cbVTK::xmlname = "VTK";

#include "../HandlerFactory.h"
#include "../utils.h"
#include "../vtkLattice.h"

int cbVTK::Init() {
    Callback::Init();

    // xml init is common for all lattice types
    pugi::xml_attribute attr = node.attribute("name");
    nm = "VTK";
    if (attr) nm = attr.value();
    attr = node.attribute("what");
    if (attr) {
        s.add_from_string(attr.value(), ',');
    } else {
        s.add_from_string("all", ',');
    }

    const auto init_cartesian = [&](const Lattice<CartLattice>* lattice) {
        const auto& global_region = lattice->connectivity.global_region;
        reg = global_region;

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
        if (reg.nx < 0) { reg.nx = global_region.nx - reg.dx + reg.nx; }
        attr = node.attribute("ny");
        if (attr) { reg.ny = solver->units.alt(attr.value()); }
        if (reg.ny < 0) { reg.ny = global_region.ny - reg.dy + reg.nz; }
        attr = node.attribute("nz");
        if (attr) { reg.nz = solver->units.alt(attr.value()); }
        if (reg.nz < 0) { reg.nz = global_region.nz - reg.dz + reg.nz; }

        reg = reg.intersect(global_region);

        debug1("VTK \"%s\" with output region: %dx%dx%d + %d,%d,%d from total region %dx%dx%d + %d,%d,%d", nm.c_str(), reg.nx, reg.ny, reg.nz, reg.dx, reg.dy, reg.dz, global_region.nx, global_region.ny, global_region.nz, global_region.dx, global_region.dy, global_region.dz);
        if (reg.size() == 0) {
            ERROR("VTK \"%s\" output has size 0", nm.c_str());
            return EXIT_FAILURE;
        }

        return EXIT_SUCCESS;
    };
    const auto init_arbitrary = [&](const Lattice<ArbLattice>* lattice) {
        /// TODO
        return EXIT_SUCCESS;
    };
    return std::visit(OverloadSet{init_cartesian, init_arbitrary}, solver->getLatticeVariant());
}

int cbVTK::DoIt() {
    Callback::DoIt();
    const auto do_cartesian = [&](Lattice<CartLattice>* lattice) {
        const auto filename = solver->outIterFile(nm, ".vti");
        return vtkWriteLattice(filename, *lattice, solver->units, s, lattice->connectivity.global_region);
    };
    const auto do_arbitrary = [&](Lattice<ArbLattice>* lattice) {
        const auto filename = solver->outIterFile(nm, ".vtu");
        return vtuWriteLattice(filename, *lattice, solver->units, s);
    };
    return std::visit(OverloadSet{do_cartesian, do_arbitrary}, solver->getLatticeVariant());
};

// Register the handler (basing on xmlname) in the Handler Factory
template class HandlerFactory::Register<GenericAsk<cbVTK> >;
