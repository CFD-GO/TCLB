#include "cbAveraging.h"
std::string cbAveraging::xmlname = "Average";
#include "../HandlerFactory.h"

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


// Register the handler (basing on xmlname) in the Handler Factory
template class HandlerFactory::Register< GenericAsk< cbAveraging > >;
