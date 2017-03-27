#include "acSAdjoint.h"
#include "../HandlerFactory.h"

int acSAdjoint::Init () {
		GenericAction::Init();
		old_iter_type = solver->iter_type;
		solver->iter_type |= ITER_LASTGLOB;
		GenericAction::ExecuteInternal();
		everyIter = std::max(everyIter,  (double) (solver->iter - startIter));
		if (everyIter <= 0) {
			WARNING("Warning: Zero iterations in steady adjoint. somethings is probaby wrong\n");
			return 0;
		}
		solver->iter_type = (old_iter_type & (~ITER_TYPE)) | ITER_ADJOINT;
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
					solver->hands[i].DoIt();
				}
			}
		} while (!Now(solver->iter));
		CudaThreadSynchronize();
		MPI_Barrier(MPI_COMM_WORLD);
		GenericAction::Unstack();
		solver->iter_type = old_iter_type;
		return 0;
	}


// Already registered by acUSAdjoint

