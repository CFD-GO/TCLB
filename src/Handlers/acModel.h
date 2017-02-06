#ifndef ACMODEL_H
#define ACMODEL_H

#include "../CommonHandler.h"

#include "vHandler.h"
#include "Action.h"
#include "GenericAction.h"
#include "GenericContainer.h"

class  acModel  : public  GenericContainer  {
	public:
	static std::string xmlname;
int Init ();
};

#endif // ACMODEL_H
