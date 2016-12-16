#include "MainContainer.h"

int MainContainer::Init () {
		GenericAction::Init();
		char filename[STRING_LEN];

        const int ret =  GenericAction::ExecuteInternal();

		solver->outIterFile("config", ".xml", filename);
		pugi::xml_node n = solver->configfile.child("CLBConfig").append_child("Run");
		n.append_attribute("model").set_value(MODEL);
		pugi::xml_node c = n.append_child("Code");
		c.append_attribute("version").set_value(VERSION);
		#ifdef CALC_DOUBLE_PRECISION
			c.append_attribute("precision").set_value("double");
		#else
			c.append_attribute("precision").set_value("float");
		#endif
		#ifdef CROSS_CPU
			c.append_attribute("cross").set_value("CPU");
		#else
			c.append_attribute("cross").set_value("GPU");
		#endif
		solver->configfile.save_file(filename);

		return ret;
	}


int MainContainer::Finish () {
		GenericAction::Unstack();
		return GenericAction::Finish();
	}

