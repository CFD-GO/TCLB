#include "cbSaveBinary.h"
std::string cbSaveBinary::xmlname = "SaveBinary";
#include "../HandlerFactory.h"

int cbSaveBinary::Init () {
		Callback::Init();
		pugi::xml_attribute attr = node.attribute("file");
		if (!attr) {
			attr = node.attribute("filename");
			if (!attr) {
				fn = solver->outIterFile("Save", "");
			} else {
                fn = attr.value();
            }
		} else {
            fn = ((std::string) solver->outpath) + "_" + attr.value();
        }
		return 0;
	}


int cbSaveBinary::DoIt () {
		Callback::DoIt();
		pugi::xml_attribute attr = node.attribute("comp");
		if (attr) {
			error("SaveBinary with selected component was not implemented");
			return -1;
		} else {
			solver->lattice->saveSolution(fn);
            	//error("Missing comp attribute in SaveBinary");
		}
		return 0;
	};


// Register the handler (basing on xmlname) in the Handler Factory
template class HandlerFactory::Register< GenericAsk< cbSaveBinary > >;
