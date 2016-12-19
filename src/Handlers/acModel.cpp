#include "acModel.h"
#include "../HandlerFactory.h"

int acModel::Init () {
		GenericContainer::Init();
		solver->lattice->Init();
		solver->iter = 0;
		return 0;
	}


// Function created only to check to create Handler for specific conditions
vHandler * Ask_For_acModel(const pugi::xml_node& node) {
  std::string name = node.name();
  if (name == "Model") {
		return new acModel;
  }
  return NULL;
}

// Register this function in the Handler Factory
template class HandlerFactory::Register< Ask_For_acModel >;

