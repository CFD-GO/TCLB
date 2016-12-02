#ifndef CBBIN_H
#define CBBIN_H

#include "../CommonHandler.h"

#include "vHandler.h"
#include "Callback.h"

class  cbBIN  : public  Callback  {
	std::string nm;
	public:
int Init ();
int DoIt ();
};

#endif // CBBIN_H
