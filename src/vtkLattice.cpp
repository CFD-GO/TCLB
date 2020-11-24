#include <stdio.h>
#include <assert.h>
#include <mpi.h>
#include "cross.h"
#include "vtkLattice.h"
#include "Global.h"

int vtkWriteLattice(char * filename, LatticeBase * lattice, UnitEnv units, name_set * what)
{
	size_t size;
	lbRegion reg = lattice->region;
	size = reg.size();
	vtkFileOut vtkFile(MPMD.local);
	if (vtkFile.Open(filename)) {return -1;}
	double spacing = 1/units.alt("m");	
	vtkFile.Init(lattice->mpi.totalregion, reg, "Scalars=\"rho\" Vectors=\"velocity\"", spacing);

	{	big_flag_t * NodeType = new big_flag_t[size];
		lattice->GetFlags(reg, NodeType);
		if (what->explicitlyIn("flag")) {
			vtkFile.WriteField("flag",NodeType);
		}
		unsigned char * small = new unsigned char[size];
		for (ModelBase::NodeTypeGroupFlags::const_iterator it = lattice->model->nodetypegroupflags.begin(); it != lattice->model->nodetypegroupflags.end(); it++) {
			if ((what->all && it->isSave) || what->explicitlyIn(it->name)) {
				for (size_t i=0;i<size;i++) {
					small[i] = (NodeType[i] & it->flag) >> it->shift;
				}
				vtkFile.WriteField(it->name.c_str(),small);
			}
		}
		delete[] small;
		delete[] NodeType;
	}

	for (ModelBase::Quantities::const_iterator it = lattice->model->quantities.begin(); it != lattice->model->quantities.end(); it++) {
		if (what->in(it->name)) {
			double v = units.alt(it->unit);
			int comp = 1;
			if (it->isVector) comp = 3;
	                real_t* tmp = new real_t[size*comp];
                        lattice->GetQuantity(it->id, reg, tmp, 1/v);
			vtkFile.WriteField(it->name.c_str(), tmp, comp);
			delete[] tmp;
		}
	}
	
	vtkFile.Finish();
	vtkFile.Close();
	return 0;
}

/**
 * Function to write lattice data out as PolyData - set of coordinates and pointdata values
 * 
 **/
int vtkWriteLatticeArbitrary(char * filename, size_t latticeSize, LatticeBase * lattice, UnitEnv units, name_set * what)
{
	size_t size;
	lbRegion reg = lattice->region;
	size = latticeSize;
	vtpFileOut vtpFile(MPMD.local);
	if (vtpFile.Open(filename)) {return -1;}
	double spacing = 1/units.alt("m");	
	vtpFile.Init(lattice->mpi.totalregion, reg, latticeSize, "Scalars=\"rho\" Vectors=\"velocity\"", spacing);

	{	big_flag_t * NodeType = new big_flag_t[size];
		lattice->GetFlags(reg, NodeType);
		if (what->explicitlyIn("flag")) {
			vtpFile.WriteField("flag",NodeType);
		}
		unsigned char * small = new unsigned char[size];
		for (ModelBase::NodeTypeGroupFlags::const_iterator it = lattice->model->nodetypegroupflags.begin(); it != lattice->model->nodetypegroupflags.end(); it++) {
			if ((what->all && it->isSave) || what->explicitlyIn(it->name)) {
				for (size_t i=0;i<size;i++) {
					small[i] = (NodeType[i] & it->flag) >> it->shift;
				}
				vtpFile.WriteField(it->name.c_str(),small);
			}
		}
		delete[] small;
		delete[] NodeType;
	}

	for (ModelBase::Quantities::const_iterator it = lattice->model->quantities.begin(); it != lattice->model->quantities.end(); it++) {
		if (what->in(it->name)) {
			double v = units.alt(it->unit);
			int comp = 1;
			if (it->isVector) comp = 3;
	                real_t* tmp = new real_t[size*comp];
                        lattice->GetQuantity(it->id, reg, tmp, 1/v);
			vtpFile.WriteField(it->name.c_str(), tmp, comp);
			delete[] tmp;
		}
	}
	
	vtpFile.FinishCellData();
	vtpFile.WritePointsHeader();
	
	real_t* temp = new real_t[size * 3];
	lattice->GetCoords(temp);

	vtpFile.WriteField("coords", temp, 3);
	delete[] temp;
	vtpFile.Finish();

	vtpFile.Close();
	return 0;
}

/**
 * Function to write data out as an unstructured grid - set of points, cells defined as edges of points, celldata
 * 
 **/
int vtkWriteLatticeArbitraryUG(char * filename, size_t latticeSize, LatticeBase * lattice, Connectivity * connectivity, UnitEnv units, name_set * what)
{
	
	size_t size;
	lbRegion reg = lattice->region;
	size = latticeSize;
	vtuFileOut vtuFile(MPMD.local);
	if (vtuFile.Open(filename)) {return -1;}
	double spacing = 1/units.alt("m");	
	vtuFile.Init(lattice->mpi.totalregion, reg, connectivity->nPoints, connectivity->nCells, "Scalars=\"rho\" Vectors=\"velocity\"", spacing);

	{	big_flag_t * NodeType = new big_flag_t[size];
		lattice->GetFlags(reg, NodeType);
		if (what->explicitlyIn("flag")) {
			vtuFile.WriteField("flag",NodeType);
		}
		unsigned char * small = new unsigned char[size];
		for (ModelBase::NodeTypeGroupFlags::const_iterator it = lattice->model->nodetypegroupflags.begin(); it != lattice->model->nodetypegroupflags.end(); it++) {
			if ((what->all && it->isSave) || what->explicitlyIn(it->name)) {
				for (size_t i=0;i<size;i++) {
					small[i] = (NodeType[i] & it->flag) >> it->shift;
				}
				vtuFile.WriteField(it->name.c_str(),small);
			}
		}
		delete[] small;
		delete[] NodeType;
	}

	for (ModelBase::Quantities::const_iterator it = lattice->model->quantities.begin(); it != lattice->model->quantities.end(); it++) {
		if (what->in(it->name)) {
			double v = units.alt(it->unit);
			int comp = 1;
			if (it->isVector) comp = 3;
	                real_t* tmp = new real_t[size*comp];
                        lattice->GetQuantity(it->id, reg, tmp, 1/v);
			vtuFile.WriteField(it->name.c_str(), tmp, comp);
			delete[] tmp;
		}
	}
	
	vtuFile.FinishCellData();
	vtuFile.WritePointsHeader();
	vtuFile.WriteField("Points", (void*)connectivity->pointData, sizeof(real_t)*3, connectivity->nPoints, "Float64", 3);
	vtuFile.WritePointsFooter();
	vtuFile.WriteCellsHeader();
	vtuFile.WriteField("connectivity", (void*)connectivity->cellConnectivity, sizeof(size_t), connectivity->nCells * 8, "Int64", 1);
	vtuFile.WriteField("offsets", (void*)connectivity->cellOffsets, sizeof(size_t), connectivity->nCells, "Int64", 1);
	vtuFile.WriteField("types", (void*)connectivity->cellTypes, sizeof(unsigned char), connectivity->nCells, "UInt8", 1);
	vtuFile.WriteCellsFooter();
	vtuFile.Finish();
	vtuFile.Close();

	return 0;
}

int binWriteLattice(char * filename, LatticeBase * lattice, UnitEnv units)
{
	int size;
	lbRegion reg = lattice->region;
	FILE * f;
	char fn[STRING_LEN];
	size = reg.size();
	for (ModelBase::Quantities::const_iterator it = lattice->model->quantities.begin(); it != lattice->model->quantities.end(); it++) {
		#ifndef ADJOINT
			if (it->isAdjoint) continue;
		#endif
		int comp = 1;
		if (it->isVector) comp = 3;
		real_t* tmp = new real_t[size*comp];
		lattice->GetQuantity(it->id, reg, tmp, 1);
		sprintf(fn, "%s.%s.bin", filename, it->name.c_str());
		f = fopen(fn,"w");
		if (f == NULL) {
			ERROR("Cannot open file: %s\n",fn);
			return -1;
		}
		fwrite(tmp, sizeof(real_t)*comp, size, f);
		fclose(f);
		delete[] tmp;
	}
	return 0;
}



inline int txtWriteElement(FILE * f, float tmp) { return fprintf(f, "%.8g" , tmp); }
inline int txtWriteElement(FILE * f, double tmp) { return fprintf(f, "%.16lg" , tmp); }
//inline int txtWriteElement(FILE * f, float3 tmp) { return fprintf(f, "%g,%g,%g" , tmp.x, tmp.y, tmp.z); }
//inline int txtWriteElement(FILE * f, double3 tmp) { return fprintf(f, "%lg,%lg,%lg" , tmp.x, tmp.y, tmp.z); }
inline int txtWriteElement(FILE * f, vector_t tmp) {
	txtWriteElement(f, tmp.x);
	fprintf(f," ");
	txtWriteElement(f, tmp.y);
	fprintf(f," ");
	return txtWriteElement(f, tmp.z);
}

template <typename T> int txtWriteField(FILE * f, T * tmp, int stop, int n) 
{
	for (int i=0;i<n;i++) {
		txtWriteElement(f, tmp[i]);
		if (((i+1) % stop) == 0) fprintf(f,"\n"); else fprintf(f, " ");
	}
	return 0;
}


int txtWriteLattice(char * filename, LatticeBase * lattice, UnitEnv units, name_set * what, int type)
{
	int size;
	char fn[STRING_LEN];
	lbRegion reg = lattice->region;
	size = reg.size();
	if (D_MPI_RANK == 0) {
		sprintf(fn,"%s_info.txt",filename);
		FILE * f = fopen(fn,"w");
		if (f == NULL) {
			ERROR("Cannot open file: %s\n");
			return -1;
		}
		fprintf(f,"dx: %lg\n", 1/units.alt("m"));
		fprintf(f,"dt: %lg\n", 1/units.alt("s"));
		fprintf(f,"dm: %lg\n", 1/units.alt("kg"));
		fprintf(f,"dT: %lg\n", 1/units.alt("K"));
		fprintf(f,"size: %d\n", size);
		fprintf(f,"NX: %d\n", reg.nx);
		fprintf(f,"NY: %d\n", reg.ny);
		fprintf(f,"NZ: %d\n", reg.nz);
		fclose(f);
	}

	for (ModelBase::Quantities::const_iterator it = lattice->model->quantities.begin(); it != lattice->model->quantities.end(); it++) {
		#ifndef ADJOINT
			if (it->isAdjoint) continue;
		#endif
		if (what->in(it->name)) {
			sprintf(fn,"%s_%s.txt",filename,it->name.c_str());
			FILE * f=NULL;
			switch (type) {
			case 0:
				f = fopen(fn,"w");
				break;
			case 1:
				char com[STRING_LEN];
				sprintf(com, "gzip > %s.gz", fn);
				f = popen(com, "w");
				break;
			default:
				ERROR("Unknown type in txtWriteLattice\n");
			}
			if (f == NULL) {
				ERROR("Cannot open file: %s\n",fn);
				return -1;
			}
			double v = units.alt(it->unit);
			real_t* tmp = new real_t[size];
			lattice->GetQuantity(it->id, reg, tmp, 1/v);
			txtWriteField(f, tmp, reg.nx, size);
			delete[] tmp;
			fclose(f);
		}
	}

	return 0;
}
