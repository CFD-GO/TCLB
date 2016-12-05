#ifndef ACSADJOINT_H
#define ACSADJOINT_H

#include "../CommonHandler.h"

#include "vHandler.h"
#include "Action.h"
#include "GenericAction.h"

class  acSAdjoint  : public  GenericAction  {
	int old_iter_type;
	public:
int Init ();
};

#endif // ACSADJOINT_H
