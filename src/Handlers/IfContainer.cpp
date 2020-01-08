#include "../HandlerFactory.h"
#include "IfContainer.h"
std::string IfContainer::xmlname = "EvalIf";

int IfContainer::Init () {
		GenericAction::Init();
        pugi::xml_attribute present = node.attribute("opt_present"); 
        pugi::xml_attribute missing = node.attribute("opt_missing"); 
	std::string option;
	bool positive = false;
	if (present) {
		if (missing) {
			ERROR("Use either opt_present or opt_missing (not both) in IfContainer\n");
			return -1;
		}
		option = present.value();
		positive = true;
	} else if (missing) {
		option = present.value();
		positive = false;
	} else {
		ERROR("Use either opt_present or opt_missing (not both) in IfContainer\n");
		return -1;
	}
        ModelBase::Options::const_iterator it = solver->lattice->model->options.ByName(present.value());
        if (it == solver->lattice->model->options.end()) {
                ERROR("Unknown option in IfContainer: %s\n", present.value());
                return -1;
        }
        if (it->isActive == positive) {
            debug1("EvalIf - proceed\n");
	    return  GenericAction::ExecuteInternal();
        } else {
            debug1("EvalIf - skipped\n");
            return 0;
        }
}


int IfContainer::Finish () {
		GenericAction::Unstack();
		return GenericAction::Finish();
	}


// Register the handler (basing on xmlname) in the Handler Factory
template class HandlerFactory::Register< GenericAsk< IfContainer > >;
