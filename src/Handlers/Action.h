#ifndef ACTION_H
#define ACTION_H

#include "../CommonHandler.h"
#include "vHandler.h"

class  Action  : public  vHandler  {
public:
int DoIt ();
int Init ();
int Finish ();
int NumberOfParameters ();
int Parameters (int type, double * tab);
int Type();
};

#endif // ACTION_H
