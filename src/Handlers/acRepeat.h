#ifndef ACREPEAT_H
#define ACREPEAT_H

#include "../CommonHandler.h"

#include "vHandler.h"
#include "Action.h"
#include "GenericAction.h"

class  acRepeat  : public  GenericAction  {
	int times;
	public:
	static std::string xmlname;
int Init ();
};

#endif // ACREPEAT_H
