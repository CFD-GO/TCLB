#ifndef ACCONNECTIVITY_H
#define ACCONNECTIVITY_H

#include "../CommonHandler.h"

#include "vHandler.h"
#include "Action.h"

class  acConnectivity  : public  Action  {
	public:
	static std::string xmlname;
int Init ();
};

#endif // ACCONNECTIVITY_H
