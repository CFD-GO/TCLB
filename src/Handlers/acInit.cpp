#include "acInit.h"

int acInit::Init () {
		Action::Init();
		solver->lattice->Init();
		solver->iter = 0;
		return 0;
	}

