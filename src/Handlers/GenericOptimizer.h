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
	virtual int OptimizerInit();
	virtual int OptimizerRun();
	virtual int OptimizerExit();
int Init ();
int Execute (const double * x, double * grad, double * f);
	friend double FOptimize(unsigned int n, const double * x, double * grad, void * data);
	friend double FMaterialMore(unsigned int n, const double * x, double * grad, void * data);
	friend double FMaterialLess(unsigned int n, const double * x, double * grad, void * data);
};

#endif // GENERICOPTIMIZER_H
