#include <stdio.h>
#include <mpi.h>
#include "vtuOutput.h"
#include "vtkOutput.h"
#include <cstring>
#include <stdlib.h>

const char * vtu_field_header = "<DataArray type=\"%s\" Name=\"%s\" format=\"binary\" encoding=\"base64\" NumberOfComponents=\"%d\">\n";
const char * vtu_field_footer = "</DataArray>\n";
const char * vtu_field_parallel = "<PDataArray type=\"%s\" Name=\"%s\" format=\"binary\" encoding=\"base64\" NumberOfComponents=\"%d\"/>\n";
const char * vtu_footer       = "</Piece>\n</UnstructuredGrid>\n</VTKFile>\n";

// Error handler
#define FERR 	if (f == NULL) {fprintf(stderr, "Error: vtkOutput tried to write before opening a file\n"); return; } 


vtuFileOut::vtuFileOut(MPI_Comm comm_)
{
    f = NULL;
    fp = NULL;
    size = 0;
    comm = comm_;
};

int vtuFileOut::Open(const char* filename) {
    char* n;
    f = fopen(filename, "w");
    if(f == NULL) {
        fprintf(stderr, "Error, could not open vtu file %s\n", filename);
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
			if (strcmp(n, ".vtu") == 0) break;
			n++;
		}
		strcpy(n, ".pvtu");
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

void vtuFileOut::WriteB64(void * tab, int len) {
	FERR;
	fprintB64(f, tab, len);
};

void vtuFileOut::Init(lbRegion regiontot, lbRegion region, size_t nPoints, size_t nCells, char* selection, double spacing) {
	FERR;
	size = nCells;
    fprintf(f, "<?xml version=\"1.0\"?>\n");
	fprintf(f, "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n");
	fprintf(f, "<UnstructuredGrid>\n");
	fprintf(f, "<Piece NumberOfPoints=\"%lx\" NumberOfCells=\"%lx\">\n", nPoints, nCells);
	fprintf(f, "<PointData>\n");
	fprintf(f, "</PointData>\n");
	fprintf(f, "<CellData>\n");
	if (fp != NULL) {
        fprintf(fp, "<?xml version=\"1.0\"?>\n");
        fprintf(fp, "<VTKFile type=\"PUnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n");
        fprintf(fp, "<PUnstructuredGrid>\n");
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
			fprintf(fp, "<Piece NumberOfPoints=\"%lx\" NumberOfCells=\"%lx\" Source=\"%s\"/>\n", nPoints, nCells, buf);
		}
	}
	delete[] buf;
	if (fp != NULL) {
		fprintf(fp, "<PPointData>\n");
		fprintf(fp, "</PPointData>\n");
		fprintf(fp, "<PCellData>\n");
	}
};

void vtuFileOut::Init(lbRegion region, size_t nPoints, size_t nCells, char* selection) {
    Init(region, region, nPoints, nCells, selection);
};

void vtuFileOut::Init(int width, int height, size_t nPoints, size_t nCells) {
    Init(lbRegion(0, 0, 0, width, height, 1), nPoints, nCells, "");
};

void vtuFileOut::WriteField(const char * name, void * data, int elem, const char * tp, int components) {
	FERR;
	int len = size*elem;
	fprintf(f, vtu_field_header, tp, name, components);
	WriteB64(&len, sizeof(int));
	WriteB64(data, size*elem);
	fprintf(f, "\n");
	fprintf(f, "%s", vtu_field_footer);
	if (fp != NULL) {
		fprintf(fp, vtu_field_parallel,  tp, name, components);
	}
};

void vtuFileOut::WriteField(const char * name, void * data, int elem, int dataLength, const char * tp, int components) {
	FERR;
	int len = dataLength*elem;
	fprintf(f, vtu_field_header, tp, name, components);
	WriteB64(&len, sizeof(int));
	WriteB64(data, dataLength*elem);
	fprintf(f, "\n");
	fprintf(f, "%s", vtu_field_footer);
	if (fp != NULL) {
		fprintf(fp, vtu_field_parallel,  tp, name, components);
	}
};

void vtuFileOut::FinishCellData() {
	FERR;
	fprintf(f, "</CellData>\n");
	if (fp != NULL) {
		fprintf(fp, "</PCellData>\n");
	}
};

void vtuFileOut::WritePointsHeader() {
	FERR;
	fprintf(f, "<Points>\n");
	if(fp != NULL) {
		fprintf(fp, "<PPoints>\n");
	}
}

void vtuFileOut::WritePointsFooter() {
	FERR;
	fprintf(f, "</Points>\n");
	if(fp != NULL) {
		fprintf(fp, "</PPoints>\n");
	}
}

void vtuFileOut::WriteCellsHeader() {
	FERR;
	fprintf(f, "<Cells>\n");
	if(fp != NULL) {
		fprintf(fp, "<PCells>\n");
	}
}

void vtuFileOut::WriteCellsFooter() {
	FERR;
	fprintf(f, "</Cells>\n");
	if(fp != NULL) {
		fprintf(fp, "</PCells>\n");
	}
}

void vtuFileOut::Finish() {
	FERR;
	fprintf(f, "%s", vtu_footer);
	if (fp != NULL) {
		fprintf(fp, "</PPoints>\n</PPolyData>\n</VTKFile>\n");
	}
};

void vtuFileOut::Close() {
	FERR;
	fclose(f);
	if (fp != NULL) fclose(fp);
	f = NULL; size=0;
};