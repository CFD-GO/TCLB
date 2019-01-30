#ifndef CBAVERAGING_H
#define CBAVERAGING_H

#include "../CommonHandler.h"

#include "vHandler.h"
#include "Callback.h"

class  cbAveraging  : public  Callback  {
        public:
	static std::string xmlname;
int Init ();
int DoIt ();
int Finish ();
};

#endif // CBAVERAGING_H
