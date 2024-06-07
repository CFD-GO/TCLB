#include "acSolve.h"
std::string acSolve::xmlname = "Solve";
#include "../HandlerFactory.h"

int acSolve::Init () {
		GenericAction::Init();
		if (GenericAction::ExecuteInternal()) return -1;
		int stop=0;
		do {
			int my_next_it = Next(solver->iter);
			int next_it = my_next_it;
			for (size_t i=0; i<solver->hands.size(); i++) {
				int it  = solver->hands[i].Next(solver->iter);
				if ((it > 0) && (it < next_it)) next_it = it;
			}
			solver->steps = next_it;
			MPI_Bcast(&solver->steps, 1, MPI_INT, 0, MPMD.local);
			solver->iter += solver->steps;
			int iter_type = solver->iter_type;
			if (solver->steps == my_next_it) iter_type |= ITER_LASTGLOB;
			solver->lattice->Iterate(solver->steps, iter_type);
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
template class HandlerFactory::Register< GenericAsk< acSolve > >;
