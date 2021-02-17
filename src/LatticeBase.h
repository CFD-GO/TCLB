#ifndef LATTICEBASE_H
#define LATTICEBASE_H 1
#include "Consts.h"
#include "cross.h"
#include <vector>
#include <utility>
#include "ZoneSettings.h"
#include "SyntheticTurbulence.h"
#include "Sampler.h"
#include "RemoteForceInterface.h"
#include "BallTree.h"
#include "pinned_allocator.hpp"
#include "Lists.h"

class lbRegion;
class FTabs;
class AFTabs;
class LatticeContainer;
class LatticeContainerBase;

#define ITER_STREAM   0x000
#define ITER_NORM     0x001
#define ITER_ADJOINT  0x002
#define ITER_OPT      0x003
#define ITER_TYPE     0x003
#define ITER_NO       0x000
#define ITER_GLOBS    0x010
#define ITER_OBJ      0x020
#define ITER_INIT     0x030
#define ITER_STEADY   0x040
#define ITER_INTEG    0x070
#define ITER_LASTGLOB 0x080
#define ITER_SKIPGRAD 0x100
const int maxSnaps=33;

class LatticePromise {
public:
  lbRegion region;
  MPIInfo mpi;
  int ns;
  int latticeType; // 0 = Cartesian lattice, 1 = ArbitraryLattice
  size_t latticeSize;
};

/// Class for computations
/**
  Class for all the memory allocation, storage, calculation
  recording for unsteady adjoint etc.
*/
class LatticeBase {
private:
public:
  //LatticeContainerBase * container;
  ModelBase * model;
  ZoneSettings zSet;
  SyntheticTurbulence ST;
  Sampler *sample; //initializing sampler with zero size
  int ZoneIter;
  int Iter; ///< Iteration (Now) - "real" time of the simulation
  int Snap, aSnap; ///< Snapshot and Adjoint Snapshot number (Now)
  double* globals; ///< Table of Globals
  lbRegion region; ///< Local lattice region
  MPIInfo mpi; ///< MPI information
  typedef rfi::RemoteForceInterface< rfi::ForceCalculator, rfi::RotParticle, rfi::ArrayOfStructures, real_t, pinned_allocator<real_t> > rfi_t;
  rfi_t RFI;
  char snapFileName[STRING_LEN*2];
  virtual ~LatticeBase ();
  virtual void Color(uchar4 *) = 0;
  virtual void FlagOverwrite(big_flag_t *, lbRegion) = 0;
  virtual void CutsOverwrite(cut_t * Q, lbRegion over) = 0;
  virtual void LoadLattice(size_t* connectivity_, vector_t* coords, big_flag_t* nodeTypes, int* directionOffsets, size_t latticeSize, int Q, int ndx, int ndy, int ndz, int mindx, int mindy, int mindz) = 0;
  virtual void Init() = 0;
  //virtual void saveSolution(const char * filename) = 0;
  //virtual void loadSolution(const char * filename) = 0;
  virtual size_t sizeOfTab() = 0;
  virtual void saveToTab(real_t * tab, int snap) = 0;
  inline void saveToTab(real_t * tab) { saveToTab(tab,Snap); };
  virtual void loadFromTab(real_t * tab, int snap) = 0;
  inline void loadFromTab(real_t * tab) { loadFromTab(tab,Snap); };
  //virtual void startRecord() = 0;
  //virtual void rewindRecord() = 0;
  //virtual void stopRecord() = 0;
  virtual void clearAdjoint() = 0;
  virtual void clearDPar() = 0;
  int(* callback)(int, int, void*);
  void* callback_data;
  void Callback(int(*)(int, int, void*), void*);
  int segment_iterations;
  int total_iterations; ///< Total iteration number counter
  int callback_iter;

  // common variables moved from Lattice/ArbitraryLattice
  int Record_Iter; ///< Recorded iteration number (Now)
  int reverse_save; ///< Flag stating if recording (Now)
  int * iSnaps;
  std::vector < std::pair < int, std::pair <int, std::pair<real_t, real_t> > > > settings_record;
  unsigned int settings_i;

  // common functions moved from Lattice/ArbitraryLattice
  int getSnap(int i);
  void startRecord();
  void saveSolution(const char *filename);
  void loadSolution(const char *filename);
  void rewindRecord();
  void stopRecord();
  virtual void listTabs(int snap, bool adjSnap, int*n, size_t ** size, void *** ptr, size_t * maxsize) = 0;
  int save(int snap, bool adjSnap, const char * filename);
  int load(int snap, bool adjSnap, const char * filename);
  void push_setting(int,real_t,real_t);
  void pop_settings();
  virtual void setSetting(int i, real_t tmp) = 0;

  inline LatticeBase(int zonesettings_, int zones_) : zSet(zonesettings_, zones_) {};
  inline void MarkIteration() {
    total_iterations ++;
    if (callback) callback_iter = callback(segment_iterations, total_iterations, callback_data);
  }
  inline void FinalIteration() { 
	total_iterations = segment_iterations;
    if (callback) callback_iter = callback(segment_iterations, total_iterations, callback_data);
  }
  inline void InitialIteration(int segiter) {
    total_iterations = 0;
    segment_iterations = segiter;
    if (callback) callback_iter = callback(segment_iterations, total_iterations, callback_data);
  }
  virtual void        Iterate(int, int) = 0;
  virtual void        IterateTill(int,int) = 0;
  virtual void	RunAction(int, int) = 0;
  virtual void IterateAction(int , int , int ) = 0;
  virtual void GetFlags(lbRegion, big_flag_t *) = 0;
  virtual void GetCoords(real_t*) = 0;
  virtual void Get_Field(int, real_t * tab) = 0;
  virtual void Set_Field(int, real_t * tab) = 0;
  virtual void Get_Field_Adj(int, real_t * tab) = 0;
  virtual void GetQuantity(int quant, lbRegion over, real_t * tab, real_t scale) = 0;
  virtual void updateAllSamples() = 0;
  virtual void getGlobals(real_t * tab) = 0;
  virtual void calcGlobals() = 0;
  virtual void clearGlobals() = 0;
  virtual void clearGlobals_Adj() = 0;
  virtual double getObjective() = 0;
  virtual void resetAverage() = 0;
  virtual void SetSetting(int i, real_t val) = 0;
  virtual real_t GetSetting(int i) = 0;
  virtual void GenerateST() = 0;

  virtual int saveComp(const char*, const char*) = 0;
  virtual int loadComp(const char*, const char*) = 0;
  virtual int getComponentIntoBuffer(const char*, real_t *&, long int* , long int* ) = 0;
  virtual int loadComponentFromBuffer(const char*, real_t*) = 0;
  virtual int getQuantityIntoBuffer(const char*, real_t*&, long int*, long int*) = 0;
};

#include "Factory.h"
typedef Factory < LatticeBase, LatticePromise > LatticeFactory;

#endif
