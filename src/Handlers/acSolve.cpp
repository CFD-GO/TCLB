#include "acSolve.h"
#include "../HandlerFactory.h"

int acSolve::Init () {
		GenericAction::Init();
		if (GenericAction::ExecuteInternal()) return -1;
		int stop=0;
		do {
			int next_it = Next(solver->iter);
			for (size_t i=0; i<solver->hands.size(); i++) {
				int it  = solver->hands[i].Next(solver->iter);
				if ((it > 0) && (it < next_it)) next_it = it;
			}
			solver->steps = next_it;
			MPI_Bcast(&solver->steps, 1, MPI_INT, 0, MPI_COMM_WORLD);
			solver->iter += solver->steps;
			solver->lattice->Iterate(solver->steps, solver->iter_type);
			CudaThreadSynchronize();
			MPI_Barrier(MPI_COMM_WORLD);
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
		CudaThreadSynchronize();
		MPI_Barrier(MPI_COMM_WORLD);
		GenericAction::Unstack();
		return 0;
	}


// Function created only to check to create Handler for specific conditions
vHandler * Ask_For_acSolve(const pugi::xml_node& node) {
  std::string name = node.name();
  if (name == "Solve") {
		return new acSolve;
  }
  return NULL;
}

// Register this function in the Handler Factory
template class HandlerFactory::Register< Ask_For_acSolve >;

