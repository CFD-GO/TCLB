#include "acRunAction.h"

std::string acRunAction::xmlname = "RunAction";

int acRunAction::Init () {
		GenericAction::Init();
		pugi::xml_attribute attr;
		attr = node.attribute("name");
		std::string action_name;
		int action;
		if (attr) {
			action_name = attr.value();
		}
		if (action_name == "") {
			ERROR("Have to specify the name of the Action in RunAction");
			return -1;
		}
		const Model::Action& act = solver->lattice->model->actions.by_name(action_name);
		if (!act) {
			ERROR("R: Unknown Action");
			return -1;		
		}
		action = act.id;
		if (GenericAction::ExecuteInternal()) return -1;
		int stop=0;
		do {
			int next_it = Next(solver->iter);
			for (size_t i=0; i<solver->hands.size(); i++) {
				int it  = solver->hands[i].Next(solver->iter);
				if ((it > 0) && (it < next_it)) next_it = it;
			}
			solver->steps = next_it;
			MPI_Bcast(&solver->steps, 1, MPI_INT, 0, MPMD.local);
			solver->iter += solver->steps;
			solver->lattice->IterateAction(action, solver->steps, solver->iter_type);
			CudaDeviceSynchronize();
			MPI_Barrier(MPMD.local);
			for (size_t i=0; i<solver->hands.size(); i++) {
				if (solver->hands[i].Now(solver->iter)) {
					int ret = solver->hands[i].DoIt();
					switch (ret) {
					case ITERATION_STOP:
						stop=1;
					case 0:
						break;
					default:
						return -1;
					}
				}
			}
			if (stop) break;
		} while (!Now(solver->iter));
		CudaDeviceSynchronize();
		MPI_Barrier(MPMD.local);
		GenericAction::Unstack();
		return 0;
	}


// Register the handler (basing on xmlname) in the Handler Factory
template class HandlerFactory::Register< GenericAsk< acRunAction > >;
