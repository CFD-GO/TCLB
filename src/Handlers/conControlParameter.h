#ifndef CONCONTROLPARAMETER_H
#define CONCONTROLPARAMETER_H

#include "../CommonHandler.h"

#include "vHandler.h"
#include "Action.h"

class  conControlParameter  : public  Action  {
	public:
	static std::string xmlname;
int Init ();
};

#endif // CONCONTROLPARAMETER_H
