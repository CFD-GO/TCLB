#include "acModel.h"

int acModel::Init () {
		GenericContainer::Init();
		solver->lattice->Init();
		solver->iter = 0;
		return 0;
	}

