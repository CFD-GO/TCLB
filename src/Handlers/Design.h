#ifndef DESIGN_H
#define DESIGN_H

#include "../CommonHandler.h"

#include "vHandler.h"
#include "Callback.h"

class  Design  : public  Callback  {
public:
int DoIt ();
int Init ();
int Finish ();
int Type();
};

#endif // DESIGN_H
