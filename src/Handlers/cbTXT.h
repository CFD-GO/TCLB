#ifndef CBTXT_H
#define CBTXT_H

#include "../CommonHandler.h"

#include "vHandler.h"
#include "Callback.h"

class  cbTXT  : public  Callback  {
	std::string nm;
	name_set s;
	int txt_type;
	public:
	static std::string xmlname;
int Init ();
int DoIt ();
};

#endif // CBTXT_H
