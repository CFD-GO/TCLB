#ifndef CBAVERAGING_H
#define CBAVERAGING_H

#include "../CommonHandler.h"

#include "vHandler.h"
#include "Callback.h"

class  cbAveraging  : public  Callback  {
        public:
int Init ();
int DoIt ();
int Finish ();
};

#endif // CBAVERAGING_H
