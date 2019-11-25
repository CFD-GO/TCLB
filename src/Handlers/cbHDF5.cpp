#include "cbHDF5.h"
std::string cbHDF5::xmlname = "HDF5";
#include "../HandlerFactory.h"
#include "../hdf5Lattice.h"

int cbHDF5::Init () {
	options = 0;
	Callback::Init();
#ifdef WITH_HDF5
		pugi::xml_attribute attr = node.attribute("name");
		nm = "HDF5";
		if (attr) nm = attr.value();
		attr = node.attribute("what");
		if (attr) {
		        s.add_from_string(attr.value(),',');
                } else {
                        s.add_from_string("all",',');
                }

                bool deflate = true;
		attr = node.attribute("compress");
		if (attr) deflate = attr.as_bool();
		if (deflate) options = options | HDF5_DEFLATE;
		bool write_xdmf = true;
		attr = node.attribute("write_xdmf");
		if (attr) write_xdmf = attr.as_bool();
                if (write_xdmf) options = options | HDF5_WRITE_XDMF;
		bool point_data = false;
		attr = node.attribute("point_data");
		if (attr) point_data = attr.as_bool();
                if (point_data) options = options | HDF5_WRITE_POINT;
		attr = node.attribute("chunk");
		bool calc_double;
#ifdef CALC_DOUBLE_PRECISION
                        calc_double = true;
#else
                        calc_double = false;
#endif
		bool write_double;
		write_double = calc_double;
		attr = node.attribute("precision");
		if (attr) {
			if (strcmp(attr.value(),"double") == 0) {
				write_double = true;
			} else if (strcmp(attr.value(),"float") == 0) {
	                        write_double = false;
			} else {
				ERROR("write_double attribute should be double of false\n", attr.value());
			}
		}
		if (write_double != calc_double) {
			if (deflate) {
				NOTICE("Writing a different type then type you calculate, probably cannot be combined with deflate");
			} else {
				notice("Writing a different type then type you calculate");
			}
		}		
                if (write_double) options = options | HDF5_WRITE_DOUBLE;
		attr = node.attribute("chunk");
		if (attr) {
			ERROR("Supplying chunk size is not yet supported");
			return -1;
                } else { // Negotiate optimal chunk dimenstions
                	unsigned long int dim[3];
			dim[0] = solver->lattice->region.nz;
			dim[1] = solver->lattice->region.ny;
			dim[2] = solver->lattice->region.nx;
			for (int i = 0; i < 3; i++) {
				unsigned long int GCD, minGCD, maxGCD;
				GCD = dim[i];
				for (int j = 0; j < 500; j++) { // This is just a safty limit of iterations
					MPI_Allreduce(&GCD, &minGCD, 1, MPI_UNSIGNED_LONG, MPI_MIN, solver->mpi_comm);
					MPI_Allreduce(&GCD, &maxGCD, 1, MPI_UNSIGNED_LONG, MPI_MAX, solver->mpi_comm);
					debug1("%d %d GCD: %ld (%ld-%ld)\n", i, j,	GCD, minGCD, maxGCD);
					if (maxGCD == minGCD) break;
					GCD = GCD % minGCD;
					if (GCD == 0) GCD = minGCD;
				}
				if (maxGCD != minGCD) {
					ERROR("Parallel GCD did not work\n");
					return -1;
				}
				chunkdim[i] = GCD;
			}
			output("Negotiated HDF5 chunks: %ldx%ldx%ld[x3]\n", chunkdim[0], chunkdim[1], chunkdim[2]);
		}                
                return 0;
#else
		ERROR("No hdf5 support at configure\n");
		return -1;
#endif
	}


int cbHDF5::DoIt () {
#ifdef WITH_HDF5
		Callback::DoIt();
		return hdf5WriteLattice(nm.c_str(), solver, &s, chunkdim, options);
#else
		return -1;
#endif
};


// Register the handler (basing on xmlname) in the Handler Factory
template class HandlerFactory::Register< GenericAsk< cbHDF5 > >;
