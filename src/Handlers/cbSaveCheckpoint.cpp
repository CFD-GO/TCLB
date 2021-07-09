#include "cbSaveCheckpoint.h"
std::string cbSaveCheckpoint::xmlname = "SaveCheckpoint";
#include "../HandlerFactory.h"

int cbSaveCheckpoint::Init () {
		Callback::Init();
		/*
			Initialisation of handler checks for keep attribute 
			inside of SaveCheckpoint. Keep attribute can be used
			to save the last "x" number of checkpoints, or specify
			"all" to keep all points. 
			DEFAULT behaviour is to return only the most recent 
			checkpoint.
		*/
		pugi::xml_attribute attr = node.attribute("keep");
		if (attr) {
			if (std::string(attr.value()) == "all"){
				// Look for the keyword all and use keep = 0 as flag to store all
				keep = 0;
			} else {
				// If the attr value is not the string all, assume it is an int
				keep = attr.as_int();
			}
		} else{
			keep = 1;
		}

		return 0;
	}


int cbSaveCheckpoint::DoIt () {
		Callback::DoIt();
		/*
			Here we saveSolution to a _x.pri file where x is the MPI rank.
			If keep == 0, then we save all solutions. Otherwise, we check
			the size of the queue; less than keep then save file, else
			delete the first set into the queue
		*/
		output("writing checkpoint");
		char restartFile[2*STRING_LEN];
		char filename[2*STRING_LEN];
		std::string fileStr;
		std::string restStr;

		solver->outIterCollectiveFile("checkpoint", "", filename);
		solver->outIterCollectiveFile("restart", ".xml", restartFile);
		
		fileStr = solver->lattice->saveSolution(filename);

		myqueue.push( fileStr );
		if (D_MPI_RANK == 0 ) {
			writeRestartFile(filename, restartFile);
			restStr = restartFile;
			myqueue_rst.push( restStr );
		}

		if (keep != 0){
			if (myqueue.size() > keep) {
				// myqueue should only ever reach the size of keep
				fileStr = myqueue.front();
				remove( fileStr.c_str() ); //Takes char
				myqueue.pop();

				if (D_MPI_RANK == 0 ) {
					restStr = myqueue_rst.front();
					remove( restStr.c_str() );
					myqueue_rst.pop();
				}
			}
		}

		return 0;
	};

int cbSaveCheckpoint::writeRestartFile( const char * fn, const char * rf ) {

		// Check if a LoadBinary attribute exists
		pugi::xml_node n1 = solver->configfile.child("CLBConfig").child("LoadBinary");
		if (!n1){
			// If it doesn't exist, create it before solve
			n1 = solver->configfile.child("CLBConfig").child("Solve");
			pugi::xml_node n2 = solver->configfile.child("CLBConfig").insert_child_before("LoadBinary", n1);
			n2.append_attribute("file").set_value(fn);
		} else {
			// If it does exist, remove it and replace it with up to date file string
			n1.remove_attribute(n1.attribute("file"));
			n1.append_attribute("file").set_value(fn);	
		}

		solver->configfile.save_file( rf );

	
	return 0;
}

// Register the handler (basing on xmlname) in the Handler Factory
template class HandlerFactory::Register< GenericAsk< cbSaveCheckpoint > >;
