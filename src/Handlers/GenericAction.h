#ifndef GENERICACTION_H
#define GENERICACTION_H

#include "../CommonHandler.h"

#include "vHandler.h"
#include "Action.h"

class  GenericAction  : public  Action  {
	int stack;
	public:
int parSize;

int Init ();
int ExecuteInternal ();
int Unstack ();
int Finish ();
int NumberOfParameters ();
int Parameters (int type, double * tab);
};

#endif // GENERICACTION_H
