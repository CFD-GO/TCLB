#ifndef RUNR_H
#define RUNR_H

#include "../Consts.h"

#ifdef WITH_R

#include <RcppCommon.h>
#include <Rcpp.h>
#include <RInside.h>                            // for the embedded R via RInside

#undef Free					// Conflict of names
#undef WARNING
#undef ERROR

#endif

#include "../CommonHandler.h"

#include "vHandler.h"
#include "Callback.h"

#ifdef WITH_R

class RunR : public  Callback  {
    std::string source;
public:
    static RInside R;
    int Init ();
    int DoIt ();
};

#endif // WITH_R

#endif // RUNR_H
