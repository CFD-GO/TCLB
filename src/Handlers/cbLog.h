#ifndef CBLOG_H
#define CBLOG_H

#include "../CommonHandler.h"

#include "vHandler.h"
#include "Callback.h"

class  cbLog  : public  Callback  {
	std::string filename;
	int old_iter_type;
	public:
int Init ();
int DoIt ();
int Finish ();
};

#endif // CBLOG_H
