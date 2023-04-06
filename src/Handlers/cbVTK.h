#ifndef CBVTK_H
#define CBVTK_H

#include "../CommonHandler.h"

#include "vHandler.h"
#include "Callback.h"

class  cbVTK  : public  Callback  {
	lbRegion reg;
	std::string nm;
	name_set s;
	public:
	static std::string xmlname;
int Init ();
int DoIt ();
};

#endif // CBVTK_H
