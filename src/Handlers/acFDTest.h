#ifndef ACFDTEST_H
#define ACFDTEST_H

#include "../CommonHandler.h"

#include "vHandler.h"
#include "Action.h"
#include "GenericAction.h"
#include "GenericOptimizer.h"

class  acFDTest  : public  GenericOptimizer  {
	double * start;
	double * dx, *x;
	double * grad;
	double * lower, * upper;
	int order;
	int par_start, par_num;
	double h_min,h_max;
	int h_n;
	public:
	static std::string xmlname;
int OptimizerInit ();
int OptimizerRun ();
int OptimizerExit ();
};

#endif // ACFDTEST_H
