#ifndef ACOPTIMIZE_H
#define ACOPTIMIZE_H

#include "../CommonHandler.h"

#include "vHandler.h"
#include "Action.h"
#include "GenericAction.h"
#include "GenericOptimizer.h"
#ifdef WITH_NLOPT
        #include  <nlopt.h>
#endif

class  acOptimize  : public  GenericOptimizer  {
#ifdef WITH_NLOPT
	nlopt_opt opt;
#endif
	std::string method;
	double * start;
	public:
int OptimizerInit ();
int OptimizerRun ();
int OptimizerExit ();
};

#endif // ACOPTIMIZE_H
