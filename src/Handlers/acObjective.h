#ifndef ACPARAMS_H
#define ACPARAMS_H

#include "../CommonHandler.h"

#include "vHandler.h"
#include "Action.h"

class  acObjective  : public  Action  {
	public:
	static std::string xmlname;
int Init ();
};

#endif // ACPARAMS_H
