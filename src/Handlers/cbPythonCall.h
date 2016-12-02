#ifndef CBPYTHONCALL_H
#define CBPYTHONCALL_H

#include "../CommonHandler.h"

#include "vHandler.h"
#include "Callback.h"

class  cbPythonCall  : public  Callback  {
	std::string fn;
	public:
int Init ();
int DoIt ();
};

#endif // CBPYTHONCALL_H
