#ifndef IFCONTAINER_H
#define IFCONTAINER_H

#include "../CommonHandler.h"

#include "vHandler.h"
#include "Action.h"
#include "GenericAction.h"

class  IfContainer  : public  GenericAction  {
	public:
	static std::string xmlname;
int Init ();
int Finish ();
};

#endif // IFCONTAINER_H
