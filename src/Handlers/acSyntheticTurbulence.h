#ifndef ACSYNTHETICTURBULENCE_H
#define ACSYNTHETICTURBULENCE_H

#include "../CommonHandler.h"

#include "vHandler.h"
#include "Action.h"

class  acSyntheticTurbulence  : public  Action  {
	public:
int ReadWaveNumer (std::string name, double * var);
int Init ();
};

#endif // ACSYNTHETICTURBULENCE_H
