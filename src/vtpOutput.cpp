#include <stdio.h>
#include <mpi.h>
#include "vtpOutput.h"
#include "vtkOutput.h"
#include <cstring>
#include <stdlib.h>

const char * vtp_field_header = "<DataArray type=\"%s\" Name=\"%s\" format=\"binary\" encoding=\"base64\" NumberOfComponents=\"%d\">\n";
const char * vtp_field_footer = "</DataArray>\n";
const char * vtp_field_parallel = "<PDataArray type=\"%s\" Name=\"%s\" format=\"binary\" encoding=\"base64\" NumberOfComponents=\"%d\"/>\n";
const char * vtp_footer       = "</Points>\n</Piece>\n</PolyData>\n</VTKFile>\n";

// Error handler
#define FERR 	if (f == NULL) {fprintf(stderr, "Error: vtkOutput tried to write before opening a file\n"); return; } 


vtpFileOut::vtpFileOut(MPI_Comm comm_)
{
    f = NULL;
    fp = NULL;
    size = 0;
    comm = comm_;
};

int vtpFileOut::Open(const char* filename) {
    char* n;
    f = fopen(filename, "w");
    if(f == NULL) {
        fprintf(stderr, "Error, could not open vtp file %s\n", filename);
        return -1;
    }
    int s = strlen(filename) + 5;
    name = new char[s];
    int rank;
    MPI_Comm_rank(comm, &rank);
    if(rank == 0) {
		strcpy(name, filename);
		n=name;
		while(*n != '\0') {
			if (strcmp(n, ".vtp") == 0) break;
			n++;
		}
		strcpy(n, ".pvtp");
		fp = fopen(name,"w");
		if (fp == NULL) {fprintf(stderr, "Error: Could not open (p)vtk file %s\n", name); return -1; }
	}
	n = name;
	while(*filename != '\0')
	{
		*n = *filename;
		if (*filename == '/') n = name; else n++;
		filename++;
	}
	*n = '\0';
	s = strlen(name)+1;
	MPI_Allreduce ( &s, &name_size, 1, MPI_INT, MPI_MAX, comm );
	return 0;
};

void vtpFileOut::WriteB64(void * tab, int len) {
	FERR;
	fprintB64(f, tab, len);
};

void vtpFileOut::Init(lbRegion regiontot, lbRegion region, size_t latticeSize, char* selection, double spacing) {
	FERR;
	size = latticeSize;
    fprintf(f, "<?xml version=\"1.0\"?>\n");
	fprintf(f, "<VTKFile type=\"PolyData\" version=\"0.1\" byte_order=\"LittleEndian\">\n");
	fprintf(f, "<PolyData>\n");
	fprintf(f, "<Piece NumberOfPoints=\"%d\" NumberOfVerts=\"0\" NumberOfLines=\"0\" NumberOfStrips=\"0\" NumberOfPolys=\"0\">\n", latticeSize);
	fprintf(f, "<PointData %s>\n", selection);
	if (fp != NULL) {
        fprintf(fp, "<?xml version=\"1.0\"?>\n");
        fprintf(fp, "<VTKFile type=\"PPolyData\" version=\"0.1\" byte_order=\"LittleEndian\">\n");
        fprintf(fp, "<PPolyData>\n");
	}
	int size;
	lbRegion reg;
	MPI_Comm_size(comm, &size);
	char * buf = new char[name_size];
	for (int i=0;i<size;i++)
	{
		reg = region;
		MPI_Bcast(&reg, 6, MPI_INT, i, comm);
		strcpy(buf, name);
		MPI_Bcast(buf, name_size, MPI_CHAR, i, comm);
		if (fp != NULL) {
			fprintf(fp, "<Piece NumberOfPoints=\"%d\" NumberOfVerts=\"0\" NumberOfLines=\"0\" NumberOfStrips=\"0\" NumberOfPolys=\"0\" Source=\"%s\"/>\n", buf);
		}
	}
	delete[] buf;
	if (fp != NULL) {
		fprintf(fp, "<PPointData %s>\n", selection);
	}
};

void vtpFileOut::Init(lbRegion region, size_t latticeSize, char* selection) {
    Init(region, region, latticeSize, selection);
};

void vtpFileOut::Init(int width, int height, size_t latticeSize) {
    Init(lbRegion(0, 0, 0, width, height, 1), latticeSize, "");
};

void vtpFileOut::WriteField(const char * name, void * data, int elem, const char * tp, int components) {
	FERR;
	int len = size*elem;
	fprintf(f, vtp_field_header, tp, name, components);
	WriteB64(&len, sizeof(int));
	WriteB64(data, size*elem);
	fprintf(f, "\n");
	fprintf(f, "%s", vtp_field_footer);
	if (fp != NULL) {
		fprintf(fp, vtp_field_parallel,  tp, name, components);
	}
};

void vtpFileOut::FinishCellData() {
	FERR;
	fprintf(f, "</PointData>\n");
	if (fp != NULL) {
		fprintf(fp, "</PPointData>\n");
	}
};

void vtpFileOut::WritePointsHeader() {
	FERR;
	fprintf(f, "<Points>\n");
	if(fp != NULL) {
		fprintf(fp, "<PPoints>\n");
	}
}

void vtpFileOut::Finish() {
	FERR;
	fprintf(f, "%s", vtp_footer);
	if (fp != NULL) {
		fprintf(fp, "</PPoints>\n</PPolyData>\n</VTKFile>\n");
	}
};

void vtpFileOut::Close() {
	FERR;
	fclose(f);
	if (fp != NULL) fclose(fp);
	f = NULL; size=0;
};