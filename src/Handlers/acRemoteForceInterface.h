#ifndef ACSYNTHETICTURBULENCE_H
#define ACSYNTHETICTURBULENCE_H

#include "../CommonHandler.h"

#include "vHandler.h"
#include "Action.h"

class  acRemoteForceInterface  : public  Action  {
	public:
	static std::string xmlname;
	int Init ();
};

#endif // ACSYNTHETICTURBULENCE_H
