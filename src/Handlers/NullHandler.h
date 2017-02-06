#ifndef NULL_HANDLER_H
#define NULL_HANDLER_H

#include "vHandler.h"

class  NullHandler  : public vHandler   {
public:
int DoIt()   { return 0; };
int Init()   { return 0; };
int Finish() { return 0; };
int Type()   { return HANDLER_GENERIC; };
};

#endif // NULL_HANDLER_H

