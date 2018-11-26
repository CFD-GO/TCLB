#ifndef ACSYNTHETICTURBULENCE_H
#define ACSYNTHETICTURBULENCE_H

#include "../CommonHandler.h"

#include "vHandler.h"
#include "Action.h"

class  acRemoteForceInterface  : public  Action  {
	public:
	static std::string xmlname;
	std::string particle_type;
	std::string sim;
	double gridSpacing;
	double verletDist;
	bool xcirc;
	bool ycirc; //JM
	bool zcirc; //JM
	int Init ();
};

#endif // ACSYNTHETICTURBULENCE_H
