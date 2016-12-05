
#ifndef VTKOUTPUT_H
#include "cross.h"
#include "types.h"
#include "Region.h"
void fprintB64(FILE* f, void * tab, int len);

class vtkFileOut {
	FILE * f;
	FILE * fp;
	char * name; int name_size;
	int parallel;
	int size;
public:
	vtkFileOut ();
	int Open(const char* filename);
	void WriteB64(void * tab, int len);
	void Init(lbRegion region, char* selection);
	void Init(lbRegion, lbRegion region, char* selection, double spacing);
	inline void Init(lbRegion tot, lbRegion region, char* selection) { Init(tot, region, selection, 0.05); }
	void Init(int width, int height);
	void WriteField(char * name, void * data, int elem, char * tp, int components);
	inline void WriteField(char * name, float * data) { WriteField(name, (void*) data, sizeof(float), "Float32", 1); };
	inline void WriteField(char * name, float2 * data) { WriteField(name, (void*) data, sizeof(float2), "Float32", 2); };
	inline void WriteField(char * name, float3 * data) { WriteField(name, (void*) data, sizeof(float3), "Float32", 3); };
	inline void WriteField(char * name, double * data) { WriteField(name, (void*) data, sizeof(double), "Float64", 1); };
	inline void WriteField(char * name, double2 * data) { WriteField(name, (void*) data, sizeof(double2), "Float64", 2); };
	inline void WriteField(char * name, double3 * data) { WriteField(name, (void*) data, sizeof(double3), "Float64", 3); };
#ifndef CALC_DOUBLE_PRECISION
	inline void WriteField(char * name, vector_t * data) { WriteField(name, (void*) data, sizeof(vector_t), "Float32", 3); };
#else
	inline void WriteField(char * name, vector_t * data) { WriteField(name, (void*) data, sizeof(vector_t), "Float64", 3); };
#endif
	inline void WriteField(char * name, int * data) { WriteField(name, (void*) data, sizeof(int), "Int32", 1); };
	inline void WriteField(char * name, char * data) { WriteField(name, (void*) data, sizeof(char), "Int8", 1); };
	inline void WriteField(char * name, unsigned char * data) { WriteField(name, (void*) data, sizeof(char), "UInt8", 1); };
	inline void WriteField(char * name, short int * data) { WriteField(name, (void*) data, sizeof(short int), "Int16", 1); };
	inline void WriteField(char * name, unsigned short int * data) { WriteField(name, (void*) data, sizeof(unsigned short int), "UInt16", 1); };
	void Finish();
	void Close();
};

#endif
#define VTKOUTPUT_H 1
