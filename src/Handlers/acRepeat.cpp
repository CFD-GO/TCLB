#include "acRepeat.h"
#include "../HandlerFactory.h"

int acRepeat::Init () {
		GenericAction::Init();
		pugi::xml_attribute attr = node.attribute("Times");
		if (attr) {
			times = attr.as_int();
		} else {
			error("no Times parameter in %s\n");
			return -1;
		}
		for (int i=0; i < times; i++) {
        		if (GenericAction::ExecuteInternal()) return -1;
                }
		return 0;
	}


// Function created only to check to create Handler for specific conditions
vHandler * Ask_For_acRepeat(const pugi::xml_node& node) {
  std::string name = node.name();
  if (name == "Repeat") {
		return new acRepeat;
  }
  return NULL;
}

// Register this function in the Handler Factory
template class HandlerFactory::Register< Ask_For_acRepeat >;

