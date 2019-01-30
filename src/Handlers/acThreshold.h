#ifndef ACTHRESHOLD_H
#define ACTHRESHOLD_H

#include "../CommonHandler.h"

#include "vHandler.h"
#include "Action.h"
#include "GenericAction.h"

class  acThreshold  : public  GenericAction  {
	int par;
	int levels;
	public:
	static std::string xmlname;
int Init ();
};

#endif // ACTHRESHOLD_H
