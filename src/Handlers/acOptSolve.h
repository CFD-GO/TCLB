#ifndef ACOPTSOLVE_H
#define ACOPTSOLVE_H

#include "../CommonHandler.h"

#include "vHandler.h"
#include "Action.h"
#include "GenericAction.h"

class  acOptSolve  : public  GenericAction  {
	public:
	static std::string xmlname;
int Init ();
};

#endif // ACOPTSOLVE_H
