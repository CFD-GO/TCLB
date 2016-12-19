#include "conFieldParameter.h"
#include "../HandlerFactory.h"

int conFieldParameter::Init () {
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
vHandler * Ask_For_conFieldParameter(const pugi::xml_node& node) {
  std::string name = node.name();
  if (name == "FieldParameter") {
		return new conFieldParameter;
  }
  return NULL;
}

// Register this function in the Handler Factory
template class HandlerFactory::Register< Ask_For_conFieldParameter >;

