#include "cbLog.h"
std::string cbLog::xmlname = "Log";
#include "../HandlerFactory.h"

int cbLog::Init () {
		Callback::Init();
		pugi::xml_attribute attr = node.attribute("name");
		std::string nm = "Log";
		if (attr) nm = attr.value();
		filename = solver->outIterFile(nm, ".csv");
		solver->initLog(filename.c_str());
		old_iter_type = solver->iter_type;
		solver->iter_type |= ITER_LASTGLOB;
		return 0;
	}


int cbLog::DoIt () {
		Callback::DoIt();
		solver->writeLog(filename.c_str());
		return 0;
	}


int cbLog::Finish () {
		solver->iter_type = old_iter_type;
		return Callback::Finish();
	}


// Register the handler (basing on xmlname) in the Handler Factory
template class HandlerFactory::Register< GenericAsk< cbLog > >;
