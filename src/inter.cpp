#include "Rcpp.h"

#define CROSS_CPP

#define SETTINGS_H
#include "cuda.h"
#include "driver_types.h"
#include "cross.h"
#include "Global.h"
#include "Region.h"
#include "Lattice.h"


class LBSolver {
 public:
  lbRegion totalRegion;
  lbRegion localRegion;
  Lattice * lattice;
  LBSolver (int nx, int ny, int nz) {
   NOTICE("Starting LBSolver with dimensions: %dx%dx%d\n",nx,ny,nz);
   localRegion.nx = nx;
   localRegion.nx = ny;
   localRegion.nx = nz;
   lattice = new Lattice(localRegion);
  }
};



RCPP_MODULE(yada){
  using namespace Rcpp ;
  class_<lbRegion>( "Region")
   .constructor()
   .field("dx", &lbRegion::dx)
   .field("dy", &lbRegion::dy)
   .field("dz", &lbRegion::dz)
   .field("nx", &lbRegion::nx)
   .field("ny", &lbRegion::ny)
   .field("nz", &lbRegion::nz)
  ;
  class_<LBSolver>("LBSolver")
   .constructor<int,int,int>()
  ;
  
}
