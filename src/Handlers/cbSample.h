#ifndef CBSAMPLE_H
#define CBSAMPLE_H

#include "../CommonHandler.h"

#include "vHandler.h"
#include "Callback.h"

class  cbSample  : public  Callback  {
	name_set s;
        std::string filename; 
	public:
	static std::string xmlname;
int Init ();
int DoIt ();
int Finish ();
};

#endif // CBSAMPLE_H
