#include "cbSaveBinary.h"
std::string cbSaveBinary::xmlname = "SaveBinary";
#include "../HandlerFactory.h"

int cbSaveBinary::Init () {
		Callback::Init();
		pugi::xml_attribute attr = node.attribute("file");
		if (!attr) {
			attr = node.attribute("filename");
			if (!attr) {
				char filename[STRING_LEN];
				solver->outIterFile("Save", "", filename);
				fn = filename;
			} else {
                fn = attr.value();
            }
		} else {
            fn = ((std::string) solver->info.outpath) + "_" + attr.value();
        }
		return 0;
	}


int cbSaveBinary::DoIt () {
		Callback::DoIt();
		pugi::xml_attribute attr = node.attribute("comp");
		if (attr) {
			solver->saveComp(fn.c_str(), attr.value());
		} else {
			solver->lattice->saveSolution(fn.c_str());
            	//error("Missing comp attribute in SaveBinary");
		}
		return 0;
	};


// Register the handler (basing on xmlname) in the Handler Factory
template class HandlerFactory::Register< GenericAsk< cbSaveBinary > >;
