#ifndef ACESYSPARTICLE_H
#define ACESYSPARTICLE_H

#include "../CommonHandler.h"

#include "vHandler.h"
#include "Action.h"
#include "acRemoteForceInterface.h"

class  acESYSParticle  : public acRemoteForceInterface  {
	public:
	static std::string xmlname;
	std::string particle_type;
	std::string sim;
	double gridSpacing;
	double verletDist;
	bool xcirc;
	bool ycirc; //JM
	bool zcirc; //JM
	int Init ();
};

#endif // ACESYSPARTICLE_H
