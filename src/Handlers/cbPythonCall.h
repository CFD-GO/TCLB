#ifndef CBPYTHONCALL_H
#define CBPYTHONCALL_H

#include "../CommonHandler.h"

#include "vHandler.h"
#include "Callback.h"

class  cbPythonCall  : public  Callback  {
	std::string fn;
	public:
	static std::string xmlname;
int Init ();
int DoIt ();
};

#endif // CBPYTHONCALL_H
