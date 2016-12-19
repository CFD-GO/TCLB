#include "cbAveraging.h"
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


// Function created only to check to create Handler for specific conditions
vHandler * Ask_For_cbAveraging(const pugi::xml_node& node) {
  std::string name = node.name();
  if (name == "Average") {
		return new cbAveraging;
  }
  return NULL;
}

// Register this function in the Handler Factory
template class HandlerFactory::Register< Ask_For_cbAveraging >;

