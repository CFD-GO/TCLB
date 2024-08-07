#include "acThreshold.h"

std::string acThreshold::xmlname = "Threshold";

int acThreshold::Init () {
		double val;
		double * start = NULL;
		double * slice = NULL;
		levels = 5;
		GenericAction::Init();
		pugi::xml_attribute attr = node.attribute("Levels");
		if (attr) {
			levels = attr.as_int();
		} else {
		        levels = 5;
			WARNING("Warning: Using default (%d) Levels in %s\n", levels, node.name());
		}
		                        
		DEBUG_M;
//		par = solver->getPars();
		par = NumberOfParameters();
		DEBUG_M;
		if (solver->mpi_rank == 0) {
        		if (par == 0) {
        			ERROR( "No parameters defined!\n");
        			return -1;
        		}			
			output("Parameters: %d\n", par);
			start = new double[par];
			slice = new double[par];
		}
		DEBUG_M;
//		solver->getPar(start);
		GetParameters(start);
		DEBUG_M;
		int msg=0;
		
		const Model::Setting& it = solver->lattice->model->settings.by_name("Threshold");
		if (!it) {
			ERROR("'Threshold' is not a setting");
			return -1;
		}
		for (int i=0; i < levels; i++) {
		        double th = (1.0 * i)/(levels-1);
		        solver->lattice->SetSetting(it, th);
		        if (slice != NULL) for (int j=0;j<par;j++) slice[j]=start[j]>th ? 1.0 : 0.0;
//        		solver->setPar(slice);
			SetParameters(slice);
        		if (GenericAction::ExecuteInternal()) return -1;
                }
		return 0;
	}


// Register the handler (basing on xmlname) in the Handler Factory
template class HandlerFactory::Register< GenericAsk< acThreshold > >;
