#ifndef GENERICACTION_H
#define GENERICACTION_H

#include "../CommonHandler.h"

#include "vHandler.h"
#include "Action.h"

class  GenericAction  : public  Action  {
	int stack;
	public:

int Init ();
int ExecuteInternal ();
int Unstack ();
int Finish ();
};

#endif // GENERICACTION_H
