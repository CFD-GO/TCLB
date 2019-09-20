
#ifndef TCLBFORCEGROUPCOMMON_H
#define TCLBFORCEGROUPCOMMON_H 1

typedef char rfi_code_t;
#define MPI_RFI_CODE_T MPI_UNSIGNED_CHAR
#define RFI_CODE_HANDSHAKE 1
#define RFI_CODE_FINISH 2
#define RFI_CODE_PARTICLES 3
#define RFI_CODE_FORCES 4
#define RFI_CODE_ABORT 0xFF

#define RFI_FINISHED ((rfi_size_t) -1)

#define RFI_DATA_R 0
#define RFI_DATA_POS 1
#define RFI_DATA_VEL 4
#define RFI_DATA_ANGVEL 7
#define RFI_DATA_IN_SIZE 10

#define RFI_DATA_VOL 0
#define RFI_DATA_FORCE 1
#define RFI_DATA_MOMENT 4
#define RFI_DATA_OUT_SIZE 7

#define RFI_DATA_SIZE 10

typedef unsigned long int rfi_size_t;
#define MPI_RFI_SIZE_T MPI_UNSIGNED_LONG

typedef double rfi_real_t;
#define MPI_RFI_REAL_T MPI_DOUBLE

#endif