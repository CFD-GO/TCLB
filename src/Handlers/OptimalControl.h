#ifndef OPTIMALCONTROL_H
#define OPTIMALCONTROL_H

#include "../CommonHandler.h"

#include "vHandler.h"
#include "Callback.h"
#include "Design.h"

class  OptimalControl  : public  Design  {
	int Pars;
	int zone_number, par_index;
	int old_iter_type;
	FILE * f;
	double lower, upper;
public:
	static std::string xmlname;
int Init ();
int NumberOfParameters ();
int Parameters (int type, double * tab);
};

#endif // OPTIMALCONTROL_H
