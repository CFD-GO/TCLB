#ifndef CBSAVECHECKPOINT_H
#define CBSAVECHECKPOINT_H

#include "../CommonHandler.h"

#include "vHandler.h"
#include "Callback.h"

class  cbSaveCheckpoint  : public  Callback  {
	std::string fn;
	std::string rf;
	bool overwrite;
	public:
		static std::string xmlname;
		int Init ();
		int DoIt ();
		int writeRestartFile();
};

#endif // CBSAVECHECKPOINT_H
