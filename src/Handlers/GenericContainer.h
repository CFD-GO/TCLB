#ifndef GENERICCONTAINER_H
#define GENERICCONTAINER_H

#include "../CommonHandler.h"

#include "vHandler.h"
#include "Action.h"
#include "GenericAction.h"

class  GenericContainer  : public  GenericAction  {
	public:
	static std::string xmlname;
int Init ();
int Finish ();
};

#endif // GENERICCONTAINER_H
