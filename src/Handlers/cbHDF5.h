#ifndef CBHDF5_H
#define CBHDF5_H

#include "../CommonHandler.h"

#include "vHandler.h"
#include "Callback.h"

class  cbHDF5  : public  Callback  {
	lbRegion reg;
	std::string nm;
	name_set s;
	unsigned long int chunkdim[3];
	unsigned int options;
public:
	static std::string xmlname;
	int Init ();
	int DoIt ();
};

#endif // CBHDF5_H
