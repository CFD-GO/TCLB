#include "acObjective.h"

#include <algorithm>

std::string acObjective::xmlname = "Objective";

int acObjective::Init () {
	const auto& model = *solver->lattice->model;
	size_t n =  model.globals.size();
        std::vector<double> glob, grad(n), inObj(n, 0.);
        glob.reserve(n);
        std::copy_n(solver->lattice->globals.cbegin(), n, std::back_inserter(glob));
	MPI_Bcast(glob.data(), n, MPI_DOUBLE, 0, solver->mpi_comm);
	pugi::xml_attribute attr;
        double obj_sum = 0;
	for (const Model::Objective& obj : model.objectives) {
		attr = node.attribute(obj.name.c_str());
		if (attr) {
			double weight = attr.as_double();
			double ret;
			obj.fun(glob.data(), &ret, grad.data());
			obj_sum += weight * ret;
			for (size_t i = 0; i < n; i++) {
				inObj[i] += + weight * grad[i];
			}
		}
	}
        int zone_number = 0;
	for (const Model::Global& g : model.globals) {
            if (g.inObjId >= 0) solver->lattice->zSet.set(g.inObjId, zone_number, inObj[g.id]);
	}
	const Model::Global& obj_glob = model.globals.by_name("Objective");
	if (obj_glob) solver->lattice->globals[ obj_glob.id ] = obj_sum;
	return 0;
}

// Register the handler (basing on xmlname) in the Handler Factory
template class HandlerFactory::Register< GenericAsk< acObjective > >;
