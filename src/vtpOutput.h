#ifndef VTPOUTPUT_H
#include "cross.h"
#include "types.h"
#include "Region.h"

class vtpFileOut {
    FILE* f;
    FILE* fp;
    char* name; int name_size;
    int parallel;
    int size;
    MPI_Comm comm;
public:
    vtpFileOut(MPI_Comm comm_ = MPI_COMM_WORLD);
    int Open(const char* filename);
	void WriteB64(void * tab, int len);
	void Init(lbRegion region, size_t latticeSize, char* selection);
	void Init(lbRegion, lbRegion region, size_t latticeSize, char* selection, double spacing);
	inline void Init(lbRegion tot, lbRegion region, size_t latticeSize, char* selection) { Init(tot, region, latticeSize, selection, 0.05); }
	void Init(int width, int height, size_t latticeSize);
	void WriteField(const char * name, void * data, int elem, const char * tp, int components);
	inline void WriteField(const char * name, float * data) { WriteField(name, (void*) data, sizeof(float), "Float32", 1); };
	inline void WriteField(const char * name, float * data, int comp) { WriteField(name, (void*) data, sizeof(float)*comp, "Float32", comp); };
	inline void WriteField(const char * name, float2 * data) { WriteField(name, (void*) data, sizeof(float2), "Float32", 2); };
	inline void WriteField(const char * name, float3 * data) { WriteField(name, (void*) data, sizeof(float3), "Float32", 3); };
	inline void WriteField(const char * name, double * data) { WriteField(name, (void*) data, sizeof(double), "Float64", 1); };
	inline void WriteField(const char * name, double * data, int comp) { WriteField(name, (void*) data, sizeof(double)*comp, "Float64", comp); };
	inline void WriteField(const char * name, double2 * data) { WriteField(name, (void*) data, sizeof(double2), "Float64", 2); };
	inline void WriteField(const char * name, double3 * data) { WriteField(name, (void*) data, sizeof(double3), "Float64", 3); };
#ifndef CALC_DOUBLE_PRECISION
	inline void WriteField(const char * name, vector_t * data) { WriteField(name, (void*) data, sizeof(vector_t), "Float32", 3); };
#else
	inline void WriteField(const char * name, vector_t * data) { WriteField(name, (void*) data, sizeof(vector_t), "Float64", 3); };
#endif
	inline void WriteField(const char * name, int * data) { WriteField(name, (void*) data, sizeof(int), "Int32", 1); };
	inline void WriteField(const char * name, char * data) { WriteField(name, (void*) data, sizeof(char), "Int8", 1); };
	inline void WriteField(const char * name, unsigned char * data) { WriteField(name, (void*) data, sizeof(char), "UInt8", 1); };
	inline void WriteField(const char * name, short int * data) { WriteField(name, (void*) data, sizeof(short int), "Int16", 1); };
	inline void WriteField(const char * name, unsigned short int * data) { WriteField(name, (void*) data, sizeof(unsigned short int), "UInt16", 1); };
	inline void WriteField(const char * name, unsigned int * data) { WriteField(name, (void*) data, sizeof(unsigned int), "UInt32", 1); };
	void FinishCellData();
	void WritePointsHeader();
	void Finish();
	void Close();
};

#endif
#define VTPOUTPUT_H 1