#ifndef ACCONTAINER_H
#define ACCONTAINER_H

#include "../CommonHandler.h"

#include "vHandler.h"
#include "Action.h"
#include "GenericAction.h"

class  acContainer  : public  GenericAction  {
	int times;
	public:
	static std::string xmlname;
int Init ();
};

#endif // ACCONTAINER_H
