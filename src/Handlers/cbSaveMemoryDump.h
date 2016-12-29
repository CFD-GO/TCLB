#ifndef CBSAVEMEMORYDUMP_H
#define CBSAVEMEMORYDUMP_H

#include "../CommonHandler.h"

#include "vHandler.h"
#include "Callback.h"

class  cbSaveMemoryDump  : public  Callback  {
	std::string fn;
	public:
	static std::string xmlname;
int Init ();
int DoIt ();
};

#endif // CBSAVEMEMORYDUMP_H
