#include "GenericOptimizer.h"

int GenericOptimizer::Init () {
		GenericAction::Init();
		Pars = NumberOfParameters();
		int ret;
		if (solver->mpi_rank == 0) {
			DEBUG_M;
			ret = OptimizerInit();
		}
		MPI_Bcast( &ret, 1, MPI_INT, 0, MPI_COMM_WORLD );
		if (ret) {
			ERROR("Failed to initialize Optimizer");
			return -1;
		}
		if (solver->mpi_rank == 0) {
			ret = OptimizerRun();
                        int msg = -1;
                        MPI_Bcast( &msg, 1, MPI_INT, 0, MPI_COMM_WORLD );
		} else {
			int msg=0;
		        while (msg == 0) {
        		        MPI_Bcast( &msg, 1, MPI_INT, 0, MPI_COMM_WORLD );
        		        if (msg != 0) break;
        		        Execute(NULL,NULL,NULL);
                        }
		}
		MPI_Bcast( &ret, 1, MPI_INT, 0, MPI_COMM_WORLD );
		if (ret) {
			ERROR("Failed to run Optimizer");
			return -1;
		}
		MPI_Barrier(MPI_COMM_WORLD);
		return 0;
	}


int GenericOptimizer::Execute (const double * x, double * grad, double * f) {
		DEBUG_M;
		solver->opt_iter++;
		SetParameters(x);
		int needgrad = -1;
		if (grad == NULL) needgrad = 0;
		MPI_Bcast( &needgrad, 1, MPI_INT, 0, MPI_COMM_WORLD );
		int old_iter = solver->iter_type;
		if (! needgrad)	{
			output("No need for the gradient\n");
			solver->iter_type |= ITER_SKIPGRAD;
		}
		if (GenericAction::ExecuteInternal()) return -1;
		solver->iter_type = old_iter;
		everyIter = solver->iter - startIter;
		if (needgrad) GetGradient(grad);
		double obj = solver->lattice->getObjective();
		if (f != NULL) {
			*f = obj;
			output("Evaluated objective: %lg\n", *f);
		}
		return 0;
	}



double FOptimize(unsigned int n, const double * x, double * grad, void * data) {
	GenericOptimizer * obj = (GenericOptimizer *) data;
	assert(n == (unsigned int)obj->Pars);
	double val = NAN;
        int msg = 0;
        output("------- Optimization iteration %3d -------\n", obj->solver->opt_iter+1);
        MPI_Bcast( &msg, 1, MPI_INT, 0, MPI_COMM_WORLD );
	int ret = obj->Execute(x, grad, &val);
	if (ret) {
		ERROR("Error while executing calculations in optimize. exiting loop.\n");
		obj->OptimizerExit();
	}
	return val;
}
double FMaterialMore(unsigned int n, const double * x, double * grad, void * data) {
	GenericOptimizer * obj = (GenericOptimizer *) data;
	double material = 0.0;
	for (size_t i=0;i<n; i++) material += x[i];
	if (grad) {
        	for (size_t i=0;i<n; i++) grad[i] = 1;
        }
        output("Material %le (%le at start)\n", material, obj->material);
	return material - obj->material;
}
double FMaterialLess(unsigned int n, const double * x, double * grad, void * data) {
	GenericOptimizer * obj = (GenericOptimizer *) data;
	double material = 0.0;
	for (size_t i=0;i<n; i++) material += x[i];
	if (grad) {
        	for (size_t i=0;i<n; i++) grad[i] = -1;
        }
        output("Material %le (%le at start)\n", material, obj->material);
	return obj->material - material;
}

int GenericOptimizer::OptimizerInit() { ERROR("Called Generic Optimizer virtual Init function"); return -1; }
int GenericOptimizer::OptimizerRun() { ERROR("Called Generic Optimizer virtual Run function"); return -1; }
int GenericOptimizer::OptimizerExit() { ERROR("Called Generic Optimizer virtual Exit function"); return -1; }
