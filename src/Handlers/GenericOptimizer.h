#ifndef GENERICOPTIMIZER_H
#define GENERICOPTIMIZER_H

#include "../CommonHandler.h"

#include "vHandler.h"
#include "Action.h"
#include "GenericAction.h"

double FOptimize(unsigned int n, const double * x, double * grad, void * data);
double FMaterialMore(unsigned int n, const double * x, double * grad, void * data);
double FMaterialLess(unsigned int n, const double * x, double * grad, void * data);

class  GenericOptimizer  : public  GenericAction  {
	public:
	int Pars;
	double material;
	virtual int OptimizerInit() { ERROR("Called Generic Optimizer virtual Init function"); return -1; }
	virtual int OptimizerRun() { ERROR("Called Generic Optimizer virtual Run function"); return -1; }
	virtual int OptimizerExit() { ERROR("Called Generic Optimizer virtual Exit function"); return -1; }
int Init ();
int Execute (const double * x, double * grad, double * f);
	friend double FOptimize(unsigned int n, const double * x, double * grad, void * data);
	friend double FMaterialMore(unsigned int n, const double * x, double * grad, void * data);
	friend double FMaterialLess(unsigned int n, const double * x, double * grad, void * data);
};

#endif // GENERICOPTIMIZER_H
