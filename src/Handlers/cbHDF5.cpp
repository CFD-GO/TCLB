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

		lbRegion global_region = solver->getCartLattice()->connectivity.global_region;
		reg = global_region;
		attr = node.attribute("dx");
		if (attr) { reg.dx = solver->units.alt(attr.value()); }
		if (reg.dx < 0) {
			reg.dx = reg.nx + reg.dx;
			reg.nx = reg.nx - reg.dx;
		}
		attr = node.attribute("dy");
		if (attr) { reg.dy = solver->units.alt(attr.value()); }
		if (reg.dy < 0) {
			reg.dy = reg.ny + reg.dy;
			reg.ny = reg.ny - reg.dy;
		}
		attr = node.attribute("dz");
		if (attr) { reg.dz = solver->units.alt(attr.value()); }
		if (reg.dz < 0) {
			reg.dz = reg.nz + reg.dz;
			reg.nz = reg.nz - reg.dz;
		}
		attr = node.attribute("nx");
		if (attr) { reg.nx = solver->units.alt(attr.value()); }
		if (reg.nx < 0) { reg.nx = global_region.nx - reg.dx + reg.nx; }
		attr = node.attribute("ny");
		if (attr) { reg.ny = solver->units.alt(attr.value()); }
		if (reg.ny < 0) { reg.ny = global_region.ny - reg.dy + reg.nz; }
		attr = node.attribute("nz");
		if (attr) { reg.nz = solver->units.alt(attr.value()); }
		if (reg.nz < 0) { reg.nz = global_region.nz - reg.dz + reg.nz; }
		reg = reg.intersect(global_region);
		debug1("HDF5 \"%s\" with output region: %dx%dx%d + %d,%d,%d from total region %dx%dx%d + %d,%d,%d", nm.c_str(), 
		reg.nx,reg.ny,reg.nz,reg.dx,reg.dy,reg.dz,global_region.nx,global_region.ny,global_region.nz,global_region.dx,global_region.dy,global_region.dz);
		if (reg.size() == 0) {
			ERROR("HDF5 \"%s\" output has size 0", nm.c_str());
			return -1;
		}

		lbRegion local_reg = reg.intersect(solver->getCartLattice()->getLocalRegion());

		attr = node.attribute("chunk");
		if (attr) {
			ERROR("Supplying chunk size is not yet supported");
			return -1;
		} else { // Negotiate optimal chunk dimentions
			unsigned long int dim[3];
			dim[0] = local_reg.nz;
			dim[1] = local_reg.ny;
			dim[2] = local_reg.nx;
			for (int i = 0; i < 3; i++) {
				unsigned long int GCD, minGCD, maxGCD;
				GCD = dim[i];
				for (int j = 0; j < 500; j++) { // This is just a safty limit of iterations
					if (local_reg.size() == 0) GCD = global_region.size(); //Guarunteed to be greater than any single dimension
					MPI_Allreduce(&GCD, &minGCD, 1, MPI_UNSIGNED_LONG, MPI_MIN, solver->mpi_comm);
					if (local_reg.size() == 0) GCD = 0;
					MPI_Allreduce(&GCD, &maxGCD, 1, MPI_UNSIGNED_LONG, MPI_MAX, solver->mpi_comm);
					myprint(1,-1,"%d %d GCD: %ld (%ld-%ld)\n", i, j,	GCD, minGCD, maxGCD);
					if (maxGCD == minGCD) break;
					GCD = GCD % minGCD;
					if (GCD == 0) GCD = minGCD;
				}
				if (maxGCD != minGCD) {
					ERROR("Parallel GCD did not work\n");
					return -1;
				}
				chunkdim[i] = minGCD;
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
		return hdf5WriteLattice(nm.c_str(), solver, &s, chunkdim, options, reg);
#else
		return -1;
#endif
};


// Register the handler (basing on xmlname) in the Handler Factory
template class HandlerFactory::Register< GenericAsk< cbHDF5 > >;
