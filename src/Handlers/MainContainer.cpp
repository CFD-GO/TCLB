#include "MainContainer.h"
std::string MainContainer::xmlname = "CLBConfig";
#include "../HandlerFactory.h"

int MainContainer::Init () {
		GenericAction::Init();
		const auto filename = solver->outIterFile("config", ".xml");
		pugi::xml_node n = solver->configfile.child("CLBConfig").append_child("Run");
		n.append_attribute("model").set_value(solver->lattice->model->name.c_str());
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
		solver->configfile.save_file(filename.c_str());

		return  GenericAction::ExecuteInternal();
}


int MainContainer::Finish () {
		GenericAction::Unstack();
		return GenericAction::Finish();
	}


// Register the handler (basing on xmlname) in the Handler Factory
template class HandlerFactory::Register< GenericAsk< MainContainer > >;
