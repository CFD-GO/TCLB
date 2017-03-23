#include "NullHandler.h"
#include "../HandlerFactory.h"

// We register here all elements which result in nothing to do.

// Function created only to check to create Handler for specific conditions
vHandler * Ask_For_NullHandler(const pugi::xml_node& node) {
  std::string name = node.name();
  if (name == "Run") {
	return new NullHandler;
  }
  return NULL;
}

// Register this function in the Handler Factory
template class HandlerFactory::Register< Ask_For_NullHandler >;

