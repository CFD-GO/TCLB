#include "cbSample.h"
std::string cbSample::xmlname = "Sample";
#include "../HandlerFactory.h"

int cbSample::Init () {
	std::string nm="Sampler";
	Callback::Init();
	if (everyIter == 0) {
				error("Iteration value in sampler should not be zero");
				return -1;
			}
	pugi::xml_attribute attr=node.attribute("what");
	if (attr) {
		s.add_from_string(attr.value(),',');
	}
	else {
		s.add_from_string("all",',');
	}
	const auto init_cartesian = [&](const Lattice<CartLattice>* lattice){
		for (pugi::xml_node par = node.first_child(); par; par = par.next_sibling()) {
			if (strcmp(par.name(),"Point") == 0) {
				lbRegion loc;
				attr = par.attribute("dx");
				if (attr) {
					loc.dx = solver->units.alt(attr.value());
				}
				attr = par.attribute("dy");
				if (attr) {
					loc.dy = solver->units.alt(attr.value());
				}
				attr = par.attribute("dz");
				if (attr) {
					loc.dz = solver->units.alt(attr.value());
				}
				loc = lattice->getLocalRegion().intersect(loc);
				if (loc.nx == 1)  lattice->sample->addPoint(loc, solver->mpi_rank);
			} else {
				error("Uknown element in Sampler\n");
				return -1;
			}
		}
		filename = solver->outIterFile(nm, ".csv");
		lattice->sample->units = &solver->units;
		lattice->sample->mpi_rank = solver->mpi_rank;
		lattice->sample->Allocate(&s,startIter,everyIter);
		lattice->sample->initCSV(filename.c_str());
		return EXIT_SUCCESS;
	};
	const auto init_arbitrary = [&](const Lattice<ArbLattice>* lattice) {
      for (pugi::xml_node par = node.first_child(); par; par = par.next_sibling()) {
        if (strcmp(par.name(),"Point") == 0) {
          lbRegion loc;
          attr = par.attribute("dx");
          if (attr) {
            loc.x = solver->units.si(attr.value());
            loc.dx = solver->units.alt(attr.value());
          }
          attr = par.attribute("dy");
          if (attr) {
            loc.y = solver->units.si(attr.value());
            loc.dy = solver->units.alt(attr.value());
          }
          attr = par.attribute("dz");
          if (attr) {
            loc.z = solver->units.si(attr.value());
            loc.dz = solver->units.alt(attr.value());
          }
		  if (loc.nx == 1) lattice->sample->addPoint(loc, solver->mpi_rank);
        } else {
          error("Uknown element in Sampler\n");
          return -1;
        }
      }
      filename = solver->outIterFile(nm, ".csv");
      lattice->sample->units = &solver->units;
      lattice->sample->mpi_rank = solver->mpi_rank;
      lattice->sample->Allocate(&s,startIter,everyIter);
      lattice->sample->initCSV(filename.c_str());
      return EXIT_SUCCESS;
    };
	return std::visit(OverloadSet{init_cartesian, init_arbitrary}, solver->getLatticeVariant());
}


int cbSample::DoIt () {
	const auto do_cartesian = [&](Lattice<CartLattice>* lattice) {
      lattice->sample->writeHistory(solver->iter);
      lattice->sample->startIter = solver->iter;
      return EXIT_SUCCESS;
    };

    const auto do_arbitrary = [&](Lattice<ArbLattice>* lattice) {
      lattice->sample->writeHistory(solver->iter);
      lattice->sample->startIter = solver->iter;
      return EXIT_SUCCESS;
    };
    return std::visit(OverloadSet{do_cartesian, do_arbitrary}, solver->getLatticeVariant());
}


int cbSample::Finish () {
	const auto end_cartesian = [&](const Lattice<CartLattice>* lattice) {
        lattice->sample->Finish();
        return Callback::Finish();
    };

    const auto end_arbitrary = [&](const Lattice<ArbLattice>* lattice) {
        lattice->sample->Finish();
        return Callback::Finish();
    };
    return std::visit(OverloadSet{end_cartesian, end_arbitrary}, solver->getLatticeVariant());
}	 


// Register the handler (basing on xmlname) in the Handler Factory
template class HandlerFactory::Register< GenericAsk< cbSample > >;
