#ifndef OPTIMALCONTROLSECOND_H
#define OPTIMALCONTROLSECOND_H

#include "../CommonHandler.h"

#include "vHandler.h"
#include "Callback.h"
#include "Design.h"

class  OptimalControlSecond  : public  Design  {
	int Pars;
	int Pars2;
	int zone_number, par_index;
	int old_iter_type;
	double * tab2;
	double lower, upper;
	FILE * f;
public:
	static std::string xmlname;
int Init ();
int NumberOfParameters ();
int Parameters (int type, double * tab);
};

#endif // OPTIMALCONTROLSECOND_H
