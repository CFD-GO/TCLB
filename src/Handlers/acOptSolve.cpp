#include "acOptSolve.h"
std::string acOptSolve::xmlname = "OptSolve";
#include "../HandlerFactory.h"

int acOptSolve::Init () {
		int old_iter_type = solver->iter_type;
		GenericAction::Init();
		if (GenericAction::ExecuteInternal()) return -1;
        	solver->iter_type = (old_iter_type & (~ITER_TYPE)) | ITER_OPT;
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
		solver->iter_type = old_iter_type;
		GenericAction::Unstack();
		return 0;
	}


// Register the handler (basing on xmlname) in the Handler Factory
template class HandlerFactory::Register< GenericAsk< acOptSolve > >;
