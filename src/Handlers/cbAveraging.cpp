#include "cbAveraging.h"

int cbAveraging::Init () {
		Callback::Init();
		solver->lattice->resetAverage();
                return 0;
        }


int cbAveraging::DoIt () {
                Callback::DoIt();
                solver->lattice->resetAverage(); // reseting averages-storing densities and setting reset_iter to iter
                return 0;
        }


int cbAveraging::Finish () {
                return Callback::Finish();
        }

