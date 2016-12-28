#ifndef REPEATCONTROL_H
#define REPEATCONTROL_H

#include "../CommonHandler.h"

#include "vHandler.h"
#include "Callback.h"
#include "Design.h"

class  RepeatControl  : public  Design  {
	int Pars;
	int Pars2;
	double * tab2;
	double lower, upper;
	FILE * f;
	Handler * hand;
	bool flip;
	double flip_level;
public:
	static std::string xmlname;
int Init ();
int Finish ();
int NumberOfParameters ();
double Flip (double v, double l, int j);
int Parameters (int type, double * tab);
};

#endif // REPEATCONTROL_H
