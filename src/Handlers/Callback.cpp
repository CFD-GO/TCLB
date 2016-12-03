#include "Callback.h"

int Callback::DoIt () {
	        return 0;
	}


int Callback::Init () {
		pugi::xml_attribute attr = node.attribute("Iterations");
		startIter = solver->iter;
		if (attr) {
			double it = solver->units.alt(attr.value());
			everyIter = it;
			output("Setting callback %s at %lf iterations\n", node.name(), it);
		} else {
			everyIter = 0;
                        output("Callback %s with no Iterations attribute", node.name());
		}
		return 0;
	}


int Callback::Finish () {
		return 0;
	}
	
int Callback::Type() { return HANDLER_CALLBACK; }

