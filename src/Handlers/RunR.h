#ifndef RUNR_H
#define RUNR_H

class Solver;
#include <RcppCommon.h>
//typedef Solver* PSolver;
class PSolver {
public:
    std::string myclass;
    Solver * solver;
    inline PSolver(Solver * solver_, std::string myclass_): solver(solver_),myclass(myclass_) {};
    inline PSolver(Solver * solver_): solver(solver_),myclass("CLBSolver") {};
    
};
namespace Rcpp {
    template<> SEXP wrap(const PSolver& f);
    template<> PSolver as(SEXP sexp);
}
#include <Rcpp.h>
#include <RInside.h>                            // for the embedded R via RInside
#undef Free
#undef WARNING
#undef ERROR
#include "../CommonHandler.h"

#include "vHandler.h"
#include "Callback.h"

class  RunR  : public  Callback  {
	public:
    static RInside R;
    int Init ();
    int DoIt ();
};

#endif // RUNR_H
