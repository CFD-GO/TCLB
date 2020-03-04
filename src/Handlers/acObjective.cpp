#include "../HandlerFactory.h"

#include "acObjective.h"
std::string acObjective::xmlname = "Objective";

int acObjective::Init () {
	ModelBase * model = solver->lattice->model;
	int zone_number = 0;
	double* glob = new double[ model->globals.size() ];
	double* grad = new double[ model->globals.size() ];
	double* inObj = new double[ model->globals.size() ];
	double obj = 0;
	for (size_t i = 0; i < model->globals.size(); i++) {
		glob[i] = solver->lattice->globals[i];
		inObj[i] = 0;
	}
	MPI_Bcast(glob, model->globals.size(), MPI_DOUBLE, 0, solver->mpi_comm);
	pugi::xml_attribute attr;
	for (ModelBase::Objectives::const_iterator it=model->objectives.begin(); it!=model->objectives.end(); it++) {
		attr = node.attribute(it->name.c_str());
		if (attr) {
			double weight = attr.as_double();
			double ret;
			it->fun(glob, &ret, grad);
			obj = obj + weight * ret;
			for (size_t i = 0; i < model->globals.size(); i++) {
				inObj[i] = inObj[i] + weight * grad[i];
			}
		}
	}
	for (ModelBase::Globals::const_iterator it=model->globals.begin(); it!=model->globals.end(); it++) {
		if (it->inObjId >= 0) solver->lattice->zSet.set(it->inObjId, zone_number, inObj[it->id]);
	}
	solver->lattice->globals[ model->globals.ByName("Objective")->id ] = obj;
	delete[] glob;
	delete[] grad;
	delete[] inObj;
	return 0;
}

// Register the handler (basing on xmlname) in the Handler Factory
template class HandlerFactory::Register< GenericAsk< acObjective > >;
