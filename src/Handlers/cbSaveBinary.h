#ifndef CBSAVEBINARY_H
#define CBSAVEBINARY_H

#include "../CommonHandler.h"

#include "vHandler.h"
#include "Callback.h"

class  cbSaveBinary  : public  Callback  {
	std::string fn;
	public:
int Init ();
int DoIt ();
};

#endif // CBSAVEBINARY_H
