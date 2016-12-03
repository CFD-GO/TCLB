#include "Action.h"

int Action::DoIt () {
	        return 0;
	}


int Action::Init () {
		pugi::xml_attribute attr = node.attribute("Iterations");
		if (attr) {
			double it = solver->units.alt(attr.value());
			startIter = solver->iter;
			everyIter = it;
			if (D_MPI_RANK == 0) {
        			output("Setting action %s at %lf iterations\n", node.name(), it);
			}
		} else {
			startIter = solver->iter;
			everyIter = 0;
		}
		if (node.attribute("output")) {
			solver->setOutput(node.attribute("output").value());
		}
		return 0;
	}


int Action::Finish () {
		return 0;
	}
	int Action::Type() { return HANDLER_ACTION; }


int Action::NumberOfParameters () {
		return -1;
	};


int Action::Parameters (int type, double * tab) {
		return -1;
	};

