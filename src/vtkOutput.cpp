#include <stdio.h>
#include "mpi.h"
#include "vtkOutput.h"
#include <cstring>
#include <stdlib.h>

const char * base64char = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

void Base64char3(unsigned char * in, int len, char * out)
{
	if (len > 3) len = 3;
	int i;
	unsigned int w = 0;
	for (i = 0; i< len; i++) w = (w << 8) + (int) (in[i]);
	for (     ; i< 3;   i++) w =  w << 8;
	for (i = 0; i< 4;   i++) {
		out[3-i] = base64char[w & 0x3F];
		w = w >> 6;
	}
	for (i = len; i <3; i++) out[i+1] = '=';
}

void fprintB64(FILE* f, void * tab, int len)
{
	#define B64LineLen 76
	char buf[B64LineLen + 1];
	buf[B64LineLen] = 0;
	int i,k=0;
	for (i=0; i<len;) {
		Base64char3(((unsigned char*)tab) + i, len - i, buf + k);
		i += 3;
		k += 4;
		if (k >= B64LineLen) {
			fprintf(f,"%s",buf);
			k = 0;
		}	
	}
	buf[k] = 0;
	fprintf(f,"%s",buf);
}

// order of % arguments: width height width height
//const char * vtk_header       = "<?xml version=\"1.0\"?>\n<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\">\n<ImageData WholeExtent=\"0 %d 0 %d 0 %d\" Origin=\"0 0 0\" Spacing=\"0.005 0.005 0.005\">\n<Piece Extent=\"0 %d 0 %d 0 %d\">\n<PointData %s>\n";
// order of % arguments: datatype fieldname
const char * vtk_field_header = "<DataArray type=\"%s\" Name=\"%s\" format=\"binary\" encoding=\"base64\" NumberOfComponents=\"%d\">\n";
const char * vtk_field_footer = "</DataArray>\n";
const char * vtk_field_parallel = "<PDataArray type=\"%s\" Name=\"%s\" format=\"binary\" encoding=\"base64\" NumberOfComponents=\"%d\"/>\n";
const char * vtk_footer       = "</CellData>\n</Piece>\n</ImageData>\n</VTKFile>\n";

// Error handler
#define FERR 	if (f == NULL) {fprintf(stderr, "Error: vtkOutput tried to write before opening a file\n"); return; } 
	
// Class for writing vtk file
vtkFileOut::vtkFileOut ()
{
	f= NULL;
	fp = NULL;
	size = 0;
};

int vtkFileOut::Open(const char* filename) {
	char * n;
	f = fopen(filename,"w");
	if (f == NULL) {fprintf(stderr, "Error: Could not open vtk file %s\n", filename); return -1; } 
	int s = strlen(filename)+5;
	name = new char[s];
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	if(rank == 0) {
		strcpy(name, filename);
		n=name;
		while(*n != '\0') {
			if (strcmp(n, ".vti") == 0) break;
			n++;
		}
		strcpy(n, ".pvti");
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
	MPI_Allreduce ( &s, &name_size, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD );
	return 0;
};

void vtkFileOut::WriteB64(void * tab, int len) {
	FERR;
	fprintB64(f, tab, len);
};

void vtkFileOut::Init(lbRegion regiontot, lbRegion region, char* selection, double spacing) {
	FERR;
	size = region.size();
	fprintf(f, "<?xml version=\"1.0\"?>\n<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\">\n");
	fprintf(f, "<ImageData WholeExtent=\"%d %d %d %d %d %d\" Origin=\"0 0 0\" Spacing=\"%lg %lg %lg\">\n",
		region.dx, region.dx + region.nx,
		region.dy, region.dy + region.ny,
		region.dz, region.dz + region.nz,
		spacing,
		spacing,
		spacing
	);
	fprintf(f, "<Piece Extent=\"%d %d %d %d %d %d\">\n",
		region.dx, region.dx + region.nx,
		region.dy, region.dy + region.ny,
		region.dz, region.dz + region.nz
	);
	fprintf(f, "<CellData %s>\n", selection);
	if (fp != NULL) {
		fprintf(fp, "<?xml version=\"1.0\"?>\n<VTKFile type=\"PImageData\" version=\"0.1\" byte_order=\"LittleEndian\">\n");
		fprintf(fp, "<PImageData WholeExtent=\"%d %d %d %d %d %d\" Origin=\"0 0 0\" Spacing=\"%lg %lg %lg\">\n",
			regiontot.dx, regiontot.dx + regiontot.nx,
			regiontot.dy, regiontot.dy + regiontot.ny,
			regiontot.dz, regiontot.dz + regiontot.nz,
			spacing,
			spacing,
			spacing
		);
	}
	int size;
	lbRegion reg;
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	char * buf = new char[name_size];
	for (int i=0;i<size;i++)
	{
		reg = region;
		MPI_Bcast(&reg, 6, MPI_INT, i, MPI_COMM_WORLD);
		strcpy(buf, name);
		MPI_Bcast(buf, name_size, MPI_CHAR, i, MPI_COMM_WORLD);
		if (fp != NULL) {
			fprintf(fp, "<Piece Extent=\"%d %d %d %d %d %d\" Source=\"%s\"/>\n",
				reg.dx, reg.dx + reg.nx,
				reg.dy, reg.dy + reg.ny,
				reg.dz, reg.dz + reg.nz,
				buf
			);
		}
	}
	delete[] buf;
	if (fp != NULL) {
		fprintf(fp, "<PCellData %s>\n", selection);
	}
};

void vtkFileOut::Init(lbRegion region, char* selection) {
	Init(region, region, selection);

//	FERR;
//	size = region.size();
//	fprintf(f, vtk_header, region.nx -1, region.ny -1, region.nz -1, region.nx -1, region.ny -1, region.nz -1,  selection);
};


void vtkFileOut::Init(int width, int height) {
	Init(lbRegion(0,0,0,width,height,1),"");
};

void vtkFileOut::WriteField(char * name, void * data, int elem, char * tp, int components) {
	FERR;
	int len = size*elem;
	fprintf(f, vtk_field_header, tp, name, components);
	WriteB64(&len, sizeof(int));
	WriteB64(data, size*elem);
	fprintf(f, "\n");
	fprintf(f, "%s", vtk_field_footer);
	if (fp != NULL) {
		fprintf(fp, vtk_field_parallel,  tp, name, components);
	}
};

void vtkFileOut::Finish() {
	FERR;
	fprintf(f, "%s", vtk_footer);
	if (fp != NULL) {
		fprintf(fp, "</PCellData>\n</PImageData>\n</VTKFile>\n");
	}
};

void vtkFileOut::Close() {
	FERR;
	fclose(f);
	if (fp != NULL) fclose(fp);
	f = NULL; size=0;
};

//int main () {
//	double p [4000];
//	for (int i = 0; i< 4000; i++) p[i] = i;
//	//fprintB64(stdout, p, 4000*sizeof(int));
//	vtkFileOut f;
//	f.Open("output/vtk1.vti");
//	f.Init(40,100);
//	f.WriteField("cos", p, sizeof(int), "Float32");
//	f.WriteField("cos", p);
//	f.Finish();
//	f.Close();
//}


