#ifndef ACOPTIMIZE_H
#define ACOPTIMIZE_H

#include "../CommonHandler.h"

#include "vHandler.h"
#include "Action.h"
#include "GenericAction.h"
#include "GenericOptimizer.h"

class  acOptimize  : public  GenericOptimizer  {
	nlopt_opt opt;
	std::string method;
	double * start;
	public:
int OptimizerInit ();
int OptimizerRun ();
int OptimizerExit ();
};

#endif // ACOPTIMIZE_H
