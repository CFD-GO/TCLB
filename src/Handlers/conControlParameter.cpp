#include "conControlParameter.h"
#include "../HandlerFactory.h"

int conControlParameter::Init () {
		Action::Init();
		pugi::xml_attribute attr = node.attribute("field");
		if (!attr) {
			ERROR("No \"field\" attribute in GeometryParameter\n");
			return -1;
		}
		std::string str = attr.value();
		

		return 0;
	}


// Function created only to check to create Handler for specific conditions
vHandler * Ask_For_conControlParameter(const pugi::xml_node& node) {
  std::string name = node.name();
  if (name == "ControlParameter") {
		return new conControlParameter;
  }
  return NULL;
}

// Register this function in the Handler Factory
template class HandlerFactory::Register< Ask_For_conControlParameter >;

