#include "Design.h"

int Design::DoIt () {
		output("Design called!");
	        return 0;
	}


int Design::Init () {
		pugi::xml_attribute attr = node.attribute("Iterations");
		startIter = solver->iter;
		if (attr) {
			ERROR("Design element in xml %s shouldent have Iteration parameter!\n", node.name());
			exit(-1);
			return -1;
		} else {
			everyIter = 0;
                        output("Design %s with no Iterations attribute", node.name());
		}
		return 0;
	}


int Design::Finish () {
		return 0;
	}
	
int Design::Type() { return HANDLER_DESIGN; }


