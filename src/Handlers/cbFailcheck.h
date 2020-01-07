#ifndef CBFAILCHECK_H
#define CBFAILCHECK_H

#include "../CommonHandler.h"

#include "vHandler.h"
#include "Callback.h"

class  cbFailcheck  : public  Callback  {
	lbRegion reg;
	bool currentlyactive;
	public:
	static std::string xmlname;
int Init ();
int DoIt ();
int Finish ();
};

#endif // CBFAILCHECK_H
