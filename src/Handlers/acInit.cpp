#include "acInit.h"
#include "../HandlerFactory.h"

int acInit::Init () {
		Action::Init();
		solver->lattice->Init();
		solver->iter = 0;
		return 0;
	}


// Function created only to check to create Handler for specific conditions
vHandler * Ask_For_acInit(const pugi::xml_node& node) {
  std::string name = node.name();
  if (name == "Init") {
		return new acInit;
  }
  return NULL;
}

// Register this function in the Handler Factory
template class HandlerFactory::Register< Ask_For_acInit >;

