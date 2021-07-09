#ifndef CBSAVECHECKPOINT_H
#define CBSAVECHECKPOINT_H

#include "../CommonHandler.h"

#include "vHandler.h"
#include "Callback.h"
#include <queue>

class  cbSaveCheckpoint  : public  Callback  {
	int keep;
	std::queue<std::string> myqueue;
	std::queue<std::string> myqueue_rst;
	public:
		static std::string xmlname;
		int Init ();
		int DoIt ();
		int writeRestartFile( const char * fn, const char * rf);
};

#endif // CBSAVECHECKPOINT_H
