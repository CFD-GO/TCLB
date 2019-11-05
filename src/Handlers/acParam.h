#ifndef ACPARAM_H
#define ACPARAM_H

#include "../CommonHandler.h"

#include "vHandler.h"
#include "Action.h"

class  acParam  : public  Action  {
	public:
	static std::string xmlname;
int Init ();
};

#endif // ACPARAM_H
