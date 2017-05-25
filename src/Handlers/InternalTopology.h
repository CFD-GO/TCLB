#ifndef INTERNALTOPOLOGY_H
#define INTERNALTOPOLOGY_H

#include "../CommonHandler.h"

#include "vHandler.h"
#include "Callback.h"
#include "Design.h"

class  InternalTopology  : public  Design  {
	int Pars;
public:
	static std::string xmlname;
int Init ();
int NumberOfParameters ();
int Parameters (int type, double * tab);
};

#endif // INTERNALTOPOLOGY_H
