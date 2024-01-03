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
				if ( keep < 0) {
					// Check the user hasn't set keep to a negative value
					error("Keeping a negative no. of chckpnts not allowed, returning default behaviour.");
					keep = 1;
				}
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
		const auto filename = solver->outIterCollectiveFile("checkpoint", "");
		const auto restartFile = solver->outIterCollectiveFile("restart", ".xml");
		auto fileStr = solver->lattice->saveSolution(filename);
                std::string restStr;
		if (D_MPI_RANK == 0 ) {
			writeRestartFile(filename.c_str(), restartFile.c_str());
			restStr = restartFile;
		}
		if (keep != 0){
			myqueue.push( fileStr );
			myqueue_rst.push( restStr );
			if (myqueue.size() > (size_t) keep) {
				// myqueue should only ever reach the size of keep
				fileStr = myqueue.front();
				int rm_result = remove( fileStr.c_str() ); //Takes char
				if (rm_result != 0) error("Checkpoint file was not deleted: %s",fileStr.c_str());
				myqueue.pop();

				if (D_MPI_RANK == 0 ) {
					restStr = myqueue_rst.front();
					rm_result = remove( restStr.c_str() );
					if (rm_result != 0) error("Restart file was not deleted: %s",restStr.c_str());
					myqueue_rst.pop();
				}
			}
		}

		return 0;
	};

int cbSaveCheckpoint::writeRestartFile( const char * fn, const char * rf ) {

		pugi::xml_document restartfile;
		for (pugi::xml_node n = solver->configfile.first_child(); n; n = n.next_sibling()){
			restartfile.append_copy(n);
		}

		pugi::xml_node n1 = restartfile.child("CLBConfig").child("LoadBinary");
		if (!n1){
			// If it doesn't exist, create it before solve
			n1 = restartfile.child("CLBConfig").child("Solve");
			pugi::xml_node n2 = restartfile.child("CLBConfig").insert_child_before("LoadBinary", n1);
			n2.append_attribute("file").set_value(fn);
		} else {
			// If it does exist, remove it and replace it with up to date file string
			n1.remove_attribute(n1.attribute("file"));
			n1.append_attribute("file").set_value(fn);	
		}

		restartfile.save_file( rf );

	
	return 0;
}

// Register the handler (basing on xmlname) in the Handler Factory
template class HandlerFactory::Register< GenericAsk< cbSaveCheckpoint > >;
