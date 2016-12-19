#include "GenericContainer.h"
#include "../HandlerFactory.h"

int GenericContainer::Init () {
		GenericAction::Init();
		return GenericAction::ExecuteInternal();
	}


int GenericContainer::Finish () {
		GenericAction::Unstack();
		return GenericAction::Finish();
	}


// Function created only to check to create Handler for specific conditions
vHandler * Ask_For_GenericContainer(const pugi::xml_node& node) {
  std::string name = node.name();
  if (name == "Units") {
		return new GenericContainer;
  }
  return NULL;
}

// Register this function in the Handler Factory
template class HandlerFactory::Register< Ask_For_GenericContainer >;

