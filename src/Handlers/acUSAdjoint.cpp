#include "acUSAdjoint.h"

int acUSAdjoint::Init () {
		GenericAction::Init();
		old_iter_type = solver->iter_type;
		int skip_grad = solver->iter_type & ITER_SKIPGRAD;
		solver->iter_type = ITER_NORM | ITER_GLOBS;
		solver->lattice->startRecord();
		GenericAction::ExecuteInternal();
		everyIter = solver->iter - startIter;
		if (everyIter <= 0) {
			ERROR("No iterations done inside of Usteady Adjoint! Nothing to do\n");
			return -1;
		}
		if (skip_grad) {
			output("Skipping adjoint, as gradient is not needed");
			solver->lattice->rewindRecord();
			solver->iter -= everyIter;
		} else {
                    solver->iter_type = (old_iter_type & (~ITER_TYPE)) | ITER_ADJOINT;
                    do {
                            int next_it = Prev(solver->iter);
                            for (size_t i=0; i<solver->hands.size(); i++) {
                                    int it  = solver->hands[i].Prev(solver->iter);
                                    if ((it > 0) && (it < next_it)) next_it = it;
                            }
                            solver->steps = next_it;
                            MPI_Bcast(&solver->steps, 1, MPI_INT, 0, MPI_COMM_WORLD);
                            solver->iter -= solver->steps;
                            solver->lattice->Iterate(solver->steps, solver->iter_type);
                            CudaThreadSynchronize();
                            MPI_Barrier(MPI_COMM_WORLD);
                            for (size_t i=0; i<solver->hands.size(); i++) {
                                    if (solver->hands[i].Now(solver->iter)) {
                                            solver->hands[i].DoIt();
                                    }
                            }
                    } while (!Now(solver->iter));
		}
		solver->lattice->stopRecord();
		solver->iter += everyIter*2;
		CudaThreadSynchronize();
		MPI_Barrier(MPI_COMM_WORLD);
		GenericAction::Unstack();
		solver->iter_type = old_iter_type;
		return 0;
	}

