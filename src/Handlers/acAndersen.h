#ifndef ACANDERSEN_H
#define ACANDERSEN_H

#include "../CommonHandler.h"

#include "vHandler.h"
#include "Action.h"
#include "GenericAction.h"

class  acAndersen  : public  GenericAction  {
	int directions;
	size_t n;
	real_t** x;
	real_t** e;
	double *p;
	double skal(real_t * a, real_t * b);

	public:
	static std::string xmlname;
	int Init ();
};

#endif // ACANDERSEN_H
