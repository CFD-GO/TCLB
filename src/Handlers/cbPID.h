#ifndef CBPID_H
#define CBPID_H

#include "../CommonHandler.h"

#include "vHandler.h"
#include "Callback.h"

class  cbPID  : public  Callback  {
	int what;
	double old_err;
	double integral;
	double target;
	int setting, zone_number;
	double itime, dtime, scale;
	double DT;
	int old_iter_type;
	double sval;
public:
	static std::string xmlname;
	int Init ();
	int DoIt ();
	int Finish ();
};

#endif // CBPID_H
