#include "conFieldParameter.h"
std::string conFieldParameter::xmlname = "FieldParameter";
#include "../HandlerFactory.h"

int conFieldParameter::Init () {
		mpi_size = solver->mpi_size;
		mpi_rank = solver->mpi_rank;
		Par_sizes = new int[mpi_size];
		Par_disp = new int[mpi_size];
		pugi::xml_attribute attr = node.attribute("field");
		if (!attr) {
			ERROR("No \"field\" attribute in GeometryParameter\n");
			return -1;
		}
		{
			std::string field = attr.value();
			ModelBase::Fields::const_iterator it = solver->lattice->model->fields.ByName(field);
			if (it ==  solver->lattice->model->fields.end()) {
				ERROR("\"%s\" is not a field\n", field.c_str());
				return -1;
			}
			if (! it->isParameter) {
				ERROR("\"%s\" is not a valid parameter field\n", field.c_str());
				return -1;
			}
			field_id = it->id;
		}
		attr = node.attribute("where");
		if (attr) {
			std::string where = attr.value();
			ModelBase::NodeTypeFlags::const_iterator it = solver->lattice->model->nodetypeflags.ByName(where);
			if (it == solver->lattice->model->nodetypeflags.end()) {
				ERROR("No \"%s\" is not a valid node type\n", where.c_str());
				return -1;
			}
			flag_value = it->flag;
			flag_mask = it->group_flag;
		} else {
			flag_value = NODE_None;
			flag_mask = NODE_NONE;
		}
		attr = node.attribute("zone");
		if (attr) {
			std::string zone = attr.value();
			int zone_number = -1;
			if (solver->geometry->SettingZones.count(zone) > 0) { 
				zone_number = solver->geometry->SettingZones[zone];
			} else {
				ERROR("Unknown zone %s in %s\n", zone.c_str(), node.name());
				return -1;
			}
			flag_mask = flag_mask | NODE_SETTINGZONE;
			flag_value = flag_value | (zone_number << ZONE_SHIFT);
		}
		Pars = CalculateNumberOfParameters ();
		output("FieldParameter with %d parameters\n",Pars);
		return Design::Init();
	}

bool conFieldParameter::FlagInDesignSpace(big_flag_t flag) {
//	return (flag  & NODE_DesignSpace) != 0;
	return (flag & flag_mask) == flag_value;
}

bool conFieldParameter::InDesignSpace(size_t i) {
	return FlagInDesignSpace(solver->geometry->geom[i]);
}


int conFieldParameter::CalculateNumberOfParameters () {
	size_t n = solver->region.sizeL();
	int j=0;
	for (size_t i=0; i<n; i++) if (InDesignSpace(i)) j++;
	Par_size = j;
	printf("-- %d -- %d --\n",(int)n, (int)j);
	debug1("Par_size: %d\n",Par_size);
	MPI_Gather(&Par_size, 1, MPI_INT, Par_sizes, 1, MPI_INT, 0, MPMD.local);
	if (mpi_rank == 0) {
		int i;
		Par_disp[0] = 0;
		for (i=0; i<mpi_size-1; i++) Par_disp[i+1] = Par_disp[i] + Par_sizes[i];
		for (i=0; i<mpi_size; i++) debug2("Proc: %d Parameters: %d Disp: %d\n", i, Par_sizes[i], Par_disp[i]);
		return Par_disp[mpi_size-1] + Par_sizes[mpi_size-1];
	}
	return 0;
};

int conFieldParameter::NumberOfParameters () {
		Pars = CalculateNumberOfParameters ();
	output("FieldParameter returning %d parameters\n",Pars);
	return Pars;
};


int conFieldParameter::LocalParameters(int type, double * tab) {
	size_t n = solver->region.sizeL();
	real_t * buf = new real_t[n];

		if ((type == PAR_GET) || (type == PAR_SET)) solver->lattice->Get_Field(field_id, buf);
		if ( type == PAR_GRAD ) {
		#ifdef ADJOINT
			solver->lattice->Get_Field_Adj(field_if,buf);
		#else
			ERROR("Cannot get gradient of Field Parameter without adjoint\n");
		#endif // ADJOINT
		}

	int j=0;
	double sum=0;
	switch(type) {
	case PAR_GET:
		for (size_t i=0; i<n; i++) if (InDesignSpace(i)) {
			tab[j] = buf[i];
			j++;
		}
		break;
	case PAR_SET:
		for (size_t i=0; i<n; i++) if (InDesignSpace(i)) {
			real_t d = buf[i];
			buf[i] = tab[j];
			d = d - buf[i];
			sum += d*d;
			j++;
		}
		output("L2 norm of parameter change: %lg\n", sqrt(sum));
		break;
	case PAR_GRAD:
		for (size_t i=0; i<n; i++) if (InDesignSpace(i)) {
			tab[j] = buf[i];
			sum += buf[i]*buf[i];
			j++;
		}
		output("L2 norm of gradient: %lg\n", sqrt(sum));
		break;
	case PAR_X:
	case PAR_Y:
	case PAR_Z:
	case PAR_T:
		{
			size_t i=0;
			for (int z=0; z<solver->region.nz; z++)
			for (int y=0; y<solver->region.ny; y++)
			for (int x=0; x<solver->region.nx; x++) {
				if (InDesignSpace(i)) {
					switch(type) {
						case PAR_X: tab[j] = x; break;
						case PAR_Y: tab[j] = y; break;
						case PAR_Z: tab[j] = z; break;
						case PAR_T: tab[j] = 0; break;
					}
					j++;
				}
				i++;
			}
		}
		break;
	}
	if ( type == PAR_SET ) solver->lattice->Set_Field(field_id, buf);
	assert(j == Par_size);
	delete[] buf;
	return 0;
};

int conFieldParameter::Parameters (int type, double * tab) {
	double * ptab = new double[Par_size];
	switch(type) {
	case PAR_GET:
	case PAR_GRAD:
	case PAR_X:
	case PAR_Y:
	case PAR_Z:
	case PAR_T:
		LocalParameters(type, ptab);
		MPI_Gatherv(ptab, Par_size, MPI_DOUBLE, tab, Par_sizes, Par_disp, MPI_DOUBLE, 0, MPMD.local);
		break;
	case PAR_SET:
		MPI_Scatterv(tab, Par_sizes, Par_disp,  MPI_DOUBLE, ptab, Par_size, MPI_DOUBLE, 0, MPMD.local);
		LocalParameters(type, ptab);
		break;
	case PAR_UPPER:
		for (size_t i=0;i<Pars;i++) tab[i]=1;
		break;
	case PAR_LOWER:
		for (size_t i=0;i<Pars;i++) tab[i]=0;
		break;
	default:
		ERROR("Unknown type %d in call to Parameters in %s\n",type,node.name());
		exit(-1);
	}
	delete[] ptab;
	return 0;
};



// Register the handler (basing on xmlname) in the Handler Factory
template class HandlerFactory::Register< GenericAsk< conFieldParameter > >;
