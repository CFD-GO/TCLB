#ifndef CBCATALYST_H
#define CBCATALYST_H

#include "../CommonHandler.h"

#include "vHandler.h"
#include "Callback.h"

class  cbCatalyst  : public  Callback  {
	std::string nm;
	static int script_number;
	static bool cellData;
//	name_set s;
	public:
	static std::string xmlname;
int Init ();
int DoIt ();
int Finish ();
};

#endif // CBCATALYST_H
