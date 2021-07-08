#include "cbSaveCheckpoint.h"
std::string cbSaveCheckpoint::xmlname = "SaveCheckpoint";
#include "../HandlerFactory.h"

int cbSaveCheckpoint::Init () {
		Callback::Init();
		/*
			This component gets both the restart file (rf) name
			and the save file name (fn) for the default case. 
			If the user wants to print out checkpoints with time
			increments, then we need to use solver to get iter
			number.
		*/
		overwrite = true;
		pugi::xml_attribute attr = node.attribute("overwrite");
		if (attr) overwrite = attr.as_bool();

		if (overwrite) {
			char restartFile[2*STRING_LEN];
			char filename[2*STRING_LEN];
			solver->outGlobalFile("restart", ".xml", restartFile);
			rf = restartFile;

			solver->outGlobalFile("CheckPoint", "", filename);
			fn = filename;
			writeRestartFile();
		}
		
		return 0;
	}


int cbSaveCheckpoint::DoIt () {
		Callback::DoIt();
		/*
			Here we saveSolution to a _x.pri file where x is the MPI rank.
			Note that we write over the xml file for each checkpoint.
		*/
		if (overwrite){
			solver->lattice->saveSolution(fn.c_str());
		} else {
			fn = "CheckPoint";
			char restartFile[2*STRING_LEN];
			char filename[2*STRING_LEN];
			solver->writeSAVECHECKPOINT(fn.c_str(), filename, restartFile);
			fn = filename;
			rf = restartFile;
			writeRestartFile(); 
		}

		return 0;
	};

int cbSaveCheckpoint::writeRestartFile() {

		// Check if a LoadBinary attribute exists
		pugi::xml_node n1 = solver->configfile.child("CLBConfig").child("LoadBinary");
		if (!n1){
			// If it doesn't exist, create it before solve
			n1 = solver->configfile.child("CLBConfig").child("Solve");
			pugi::xml_node n2 = solver->configfile.child("CLBConfig").insert_child_before("LoadBinary", n1);
			n2.append_attribute("file").set_value(fn.c_str());
		} else {
			// If it does exist, remove it and replace it with up to date file string
			n1.remove_attribute(n1.attribute("file"));
			n1.append_attribute("file").set_value(fn.c_str());	
		}
		
		solver->configfile.save_file(rf.c_str());

	
	return 0;
}

// Register the handler (basing on xmlname) in the Handler Factory
template class HandlerFactory::Register< GenericAsk< cbSaveCheckpoint > >;
