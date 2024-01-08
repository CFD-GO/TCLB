#include <stdio.h>
#include <assert.h>
#include <mpi.h>
#include "cross.h"
#include "hdf5Lattice.h"
#include "Global.h"
#include "glue.hpp"
#include "mpitools.hpp"

#ifdef WITH_HDF5
	#include <hdf5.h>
#endif

std::string NameXPath(const pugi::xml_node& node) {
	std::string path;
	if (node.parent()) {
		path = NameXPath(node.parent());
		path = path + "/" + node.name();
		pugi::xml_attribute attr = node.attribute("Name");
		if (attr) {
			path = path + "[@Name='" + attr.value() + "']";
		}
	}
	return path;
}

int hdf5WriteLattice(const char * nm, Solver * solver, name_set * what, unsigned long int * chunkdim_, unsigned int options, lbRegion total_output_reg)
{
#ifdef WITH_HDF5
	Glue glue;
	CartLattice * lattice = solver->getCartLattice();
	UnitEnv * units = &solver->units;
	double unit;

	std::string filename = solver->outIterCollectiveFile(nm, ".h5");
	std::string basename = filename.substr(filename.find_last_of('/') + 1);

	size_t size;
	lbRegion local_reg = lattice->getLocalRegion();
	lbRegion reg = local_reg.intersect(total_output_reg);
	size = reg.size();

	myprint(1,-1,"Writing region %dx%dx%d + %d,%d,%d (size %d) from %dx%dx%d + %d,%d,%d",
		reg.nx,reg.ny,reg.nz,reg.dx,reg.dy,reg.dz, size,
		local_reg.nx,local_reg.ny,local_reg.nz,local_reg.dx,local_reg.dy,local_reg.dz);

	pugi::xml_document xdmf_doc;
	pugi::xml_node xdmf_main = xdmf_doc.append_child("Xdmf");
	xdmf_main.append_attribute("xmlns:xi") = "http://www.w3.org/2001/XInclude";
	xdmf_main.append_attribute("Version") = "3.0";
	pugi::xml_node xdmf_domain = xdmf_main.append_child("Domain");
	pugi::xml_node xdmf_grid = xdmf_domain.append_child("Grid");
	xdmf_grid.append_attribute("Name") = "Lattice";
	pugi::xml_node xdmf_time = xdmf_grid.append_child("Time");
	unit = units->alt("1s");
	xdmf_time.append_attribute("Value") = solver->iter / unit;
	pugi::xml_node xdmf_geometry = xdmf_grid.append_child("Geometry");
	xdmf_geometry.append_attribute("Origin") = "";
	xdmf_geometry.append_attribute("Type") = "ORIGIN_DXDYDZ";
	pugi::xml_node xdmf_dataitem;
	unit = units->alt("1m");
	xdmf_dataitem = xdmf_geometry.append_child("DataItem");
	xdmf_dataitem.append_attribute("DataType") = "Float";
	xdmf_dataitem.append_attribute("Dimensions") = "3";
	xdmf_dataitem.append_attribute("Format") = "XML";
	xdmf_dataitem.append_attribute("Precision") = 8;
	{
		double shift = 0.0;
		if (options & HDF5_WRITE_POINT) shift = 0.5;
		xdmf_dataitem.append_child(pugi::node_pcdata).set_value(glue(" ") << (lattice->pz + shift + total_output_reg.dz)/unit << (lattice->py + shift + total_output_reg.dy)/unit << (lattice->px + shift + total_output_reg.dx)/unit);
	}
	xdmf_dataitem = xdmf_geometry.append_child("DataItem");
	xdmf_dataitem.append_attribute("DataType") = "Float";
	xdmf_dataitem.append_attribute("Dimensions") = "3";
	xdmf_dataitem.append_attribute("Format") = "XML";
	xdmf_dataitem.append_attribute("Precision") = 8;
	xdmf_dataitem.append_child(pugi::node_pcdata).set_value(glue(" ") << 1/unit << 1/unit << 1/unit);

	hid_t       file_id, dset_id;         /* file and dataset identifiers */
	hsize_t     totaldim[4];                 /* dataset dimensions */
	hsize_t     dim[4];            /* local dimensions */
	hsize_t     chunkdim[4];            /* local dimensions */
	hsize_t     totalpointdim[4];            /* point dimensions */
	hsize_t	offset[4];
	int         *data;                    /* pointer to data buffer to write */
	hsize_t	ones[4];
	int         i;
	herr_t	status;
	hid_t plist_id;



	ones[0] = 1;
	ones[1] = 1;
	ones[2] = 1;
	ones[3] = 1;

	totaldim[0] = total_output_reg.nz;
	totaldim[1] = total_output_reg.ny;
	totaldim[2] = total_output_reg.nx;
	totaldim[3] = 3;
	dim[0] = reg.nz;
	dim[1] = reg.ny;
	dim[2] = reg.nx;
	dim[3] = 3;
	offset[0] = reg.dz - total_output_reg.dz;
	offset[1] = reg.dy - total_output_reg.dy;
	offset[2] = reg.dx - total_output_reg.dx;
	offset[3] = 0;

	totalpointdim[0] = total_output_reg.nz+1;
	totalpointdim[1] = total_output_reg.ny+1;
	totalpointdim[2] = total_output_reg.nx+1;


	MPI_Comm comm  = MPMD.local;
	MPI_Info info  = MPI_INFO_NULL;

	if (chunkdim_ != NULL) {
		chunkdim[0] = chunkdim_[0];
		chunkdim[1] = chunkdim_[1];
		chunkdim[2] = chunkdim_[2];
		chunkdim[3] = 3;
	} else {
		chunkdim[0] = 1;
		chunkdim[1] = 1;
		chunkdim[2] = totaldim[3];
		chunkdim[3] = totaldim[3];
	}

	pugi::xml_node xdmf_topology = xdmf_grid.append_child("Topology");
	if (options & HDF5_WRITE_POINT) {
		xdmf_topology.append_attribute("Dimensions") = glue(" ") << std::make_pair(totaldim,3);
	} else {
		xdmf_topology.append_attribute("Dimensions") = glue(" ") << std::make_pair(totalpointdim,3);
	}
	xdmf_topology.append_attribute("Type") = "3DCoRectMesh";

	pugi::xml_node xdmf_attribute;

	myprint(2,-1,"hdf5 file: %s\n   domain: %lldx%lldx%lld chunks: %lldx%lldx%lld local: %lldx%lldx%lld+%lld,%lld,%lld\n",
		filename.c_str(),
		totaldim[0], totaldim[1], totaldim[2],
		chunkdim[0], chunkdim[1], chunkdim[2],
		dim[0], dim[1], dim[2],
		offset[0], offset[1], offset[2]);

	plist_id = H5Pcreate(H5P_FILE_ACCESS);
	H5Pset_fapl_mpio(plist_id, comm, info);
	file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
	H5Pclose(plist_id);

	std::vector<big_flag_t> NodeType = lattice->getFlags(reg);
	std::vector<unsigned char> tmp;
	tmp.resize(NodeType.size());
	for (const Model::NodeTypeGroupFlag& it : lattice->model->nodetypegroupflags) {
		hid_t       filespace, memspace;
		const char * fieldname = it.name.c_str();
		hid_t input_type = H5T_NATIVE_UCHAR;
		hid_t output_type = H5T_NATIVE_UCHAR;
		int output_precision = 1;
		int rank = 3;
		filespace = H5Screate_simple(rank, totaldim, NULL);
		memspace  = H5Screate_simple(rank, dim, NULL);
		plist_id = H5Pcreate(H5P_DATASET_CREATE);


		status = H5Pset_chunk (plist_id, rank, chunkdim);
		if (status < 0) return H5Eprint1(stderr);
		if (options & HDF5_DEFLATE) status = H5Pset_deflate (plist_id, 6);
		if (status < 0) return H5Eprint1(stderr);
	   	dset_id = H5Dcreate2(file_id, fieldname, output_type, filespace, H5P_DEFAULT, plist_id, H5P_DEFAULT);
		H5Pclose(plist_id);
		if (size == 0) {
			status = H5Sselect_none(filespace);
		} else {
	  		status = H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, ones, ones, dim);
		}
	   	if (status < 0) return H5Eprint1(stderr);


		for (size_t i=0;i<size;i++) {
			tmp[i] = (NodeType[i] & it.flag) >> it.shift;
		}

		myprint(0,-1,"filespace: %lld memsize: %lld\n", H5Sget_select_npoints(filespace), H5Sget_select_npoints(memspace));
		plist_id = H5Pcreate(H5P_DATASET_XFER);
		H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
		status = H5Dwrite(dset_id, input_type, memspace, filespace, plist_id, tmp.data());

		H5Pclose(plist_id);
		H5Sclose(filespace);
		H5Sclose(memspace);
		H5Dclose(dset_id);

		xdmf_attribute = xdmf_grid.append_child("Attribute");
		if (options & HDF5_WRITE_POINT) {
			xdmf_attribute.append_attribute("Center") = "Node";
		} else {
			xdmf_attribute.append_attribute("Center") = "Cell";
		}
		xdmf_attribute.append_attribute("Name") = fieldname;
		xdmf_dataitem = xdmf_attribute.append_child("DataItem");
		xdmf_dataitem.append_attribute("DataType") = "UInt8";
		xdmf_dataitem.append_attribute("Dimensions") = glue(" ") << std::make_pair(totaldim, rank);
		xdmf_dataitem.append_attribute("Format") = "HDF";
		xdmf_dataitem.append_attribute("Precision") = output_precision;
		xdmf_dataitem.append_child(pugi::node_pcdata).set_value(glue(":") << basename << fieldname);
		std::string xdmf_dataitem_path = NameXPath(xdmf_dataitem);
	}

	for (const Model::Quantity& it : lattice->model->quantities) {
		if (what->in(it.name)) {
			hid_t       filespace, memspace;
			const char * fieldname = it.name.c_str();
			bool vector = it.isVector;
#ifdef CALC_DOUBLE_PRECISION
			hid_t input_type = H5T_NATIVE_DOUBLE;
#else
			hid_t input_type = H5T_NATIVE_FLOAT;
#endif
			hid_t output_type;
			int output_precision;
			if (options & HDF5_WRITE_DOUBLE) {
				output_type = H5T_NATIVE_DOUBLE;
				output_precision = 8;
			} else {
				output_type = H5T_NATIVE_FLOAT;
				output_precision = 4;
			}
			int rank = 3;
			if (vector) rank = 4;
			filespace = H5Screate_simple(rank, totaldim, NULL);
			memspace  = H5Screate_simple(rank, dim, NULL);
			plist_id = H5Pcreate(H5P_DATASET_CREATE);

			status = H5Pset_chunk (plist_id, rank, chunkdim);
			if (status < 0) return H5Eprint1(stderr);
			if (options & HDF5_DEFLATE) status = H5Pset_deflate (plist_id, 6);
			if (status < 0) return H5Eprint1(stderr);
			dset_id = H5Dcreate2(file_id, fieldname, output_type, filespace, H5P_DEFAULT, plist_id, H5P_DEFAULT);
			H5Pclose(plist_id);

			if (size == 0) {
				status = H5Sselect_none(filespace);
			} else {
				status = H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, ones, ones, dim);
			}
			if (status < 0) return H5Eprint1(stderr);

			unit = units->alt(it.unit);
			int comp = it.getComp();

            std::vector<real_t> tmp = lattice->getQuantity(it, reg, 1/unit);

			myprint(0,-1,"filespace: %lld memsize: %lld\n", H5Sget_select_npoints(filespace), H5Sget_select_npoints(memspace));
			plist_id = H5Pcreate(H5P_DATASET_XFER);
			H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
			status = H5Dwrite(dset_id, input_type, memspace, filespace, plist_id, tmp.data());

			H5Pclose(plist_id);
			H5Sclose(filespace);
			H5Sclose(memspace);
			H5Dclose(dset_id);

			xdmf_attribute = xdmf_grid.append_child("Attribute");
			if (options & HDF5_WRITE_POINT) {
				xdmf_attribute.append_attribute("Center") = "Node";
			} else {
				xdmf_attribute.append_attribute("Center") = "Cell";
			}
			if (vector) xdmf_attribute.append_attribute("AttributeType") = "Vector";
			xdmf_attribute.append_attribute("Name") = fieldname;
			xdmf_dataitem = xdmf_attribute.append_child("DataItem");
			xdmf_dataitem.append_attribute("DataType") = "Float";
			xdmf_dataitem.append_attribute("Dimensions") = glue(" ") << std::make_pair(totaldim, rank);
			xdmf_dataitem.append_attribute("Format") = "HDF";
			xdmf_dataitem.append_attribute("Precision") = output_precision;
			xdmf_dataitem.append_child(pugi::node_pcdata).set_value(glue(":") << basename << fieldname);
			std::string xdmf_dataitem_path = NameXPath(xdmf_dataitem);
			if (options & HDF5_WRITE_LBM) {
				xdmf_attribute = xdmf_grid.append_child("Attribute");
				if (options & HDF5_WRITE_POINT) {
					xdmf_attribute.append_attribute("Center") = "Node";
				} else {
					xdmf_attribute.append_attribute("Center") = "Cell";
				}
				if (vector) xdmf_attribute.append_attribute("AttributeType") = "Vector";
				xdmf_attribute.append_attribute("Name") = glue("_") << fieldname << "LB";
				xdmf_dataitem = xdmf_attribute.append_child("DataItem");
				xdmf_dataitem.append_attribute("ItemType") = "Function";
				xdmf_dataitem.append_attribute("Function") = glue(" ") << unit << "*" << "$0";
				xdmf_dataitem.append_attribute("Dimensions") = glue(" ") << std::make_pair(totaldim, rank);
				xdmf_dataitem = xdmf_dataitem.append_child("DataItem");
				xdmf_dataitem.append_attribute("Reference") = xdmf_dataitem_path.c_str();
			}
		}
	}

	H5Fclose(file_id);


	if (options & HDF5_WRITE_XDMF) {
		if (mpitools::MPI_Rank(comm) == 0) {
			std::string xdmf_filename = solver->outIterCollectiveFile(nm, ".xmf");
			xdmf_doc.save_file(xdmf_filename.c_str());
		}
	}

	return 0;
#else
	return -1;
#endif
}
