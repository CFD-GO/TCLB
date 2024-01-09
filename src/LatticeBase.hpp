#ifndef LATTICEBASE_HPP
#define LATTICEBASE_HPP

#include <utility>
#include <vector>

#include "Consts.h"
#include "LatticeData.hpp"
#include "Lists.h"
#include "SolidContainer.h"
#include "SyntheticTurbulence.h"
#include "ZoneSettings.h"
#include "cross.h"
#include "unit.h"

#define ITER_STREAM 0x000
#define ITER_NORM 0x001
#define ITER_ADJOINT 0x002
#define ITER_OPT 0x003
#define ITER_TYPE 0x003
#define ITER_NO 0x000
#define ITER_GLOBS 0x010
#define ITER_OBJ 0x020
#define ITER_INIT 0x030
#define ITER_STEADY 0x040
#define ITER_INTEG 0x070
#define ITER_LASTGLOB 0x080
#define ITER_SKIPGRAD 0x100

/// Base class for Lattice classes
/**
  Contains data common for all grid types, along with a common interface
*/
class LatticeBase {
    using setting_record_t = std::vector<std::pair<int, std::pair<int, std::pair<real_t, real_t> > > >;

   public:
    LatticeBase(int zonesettings, int zones, int num_snaps_, const UnitEnv& units_);
    LatticeBase(const LatticeBase&) = delete;
    LatticeBase(LatticeBase&&) = delete;
    LatticeBase& operator=(const LatticeBase&) = delete;
    LatticeBase& operator=(LatticeBase&&) = delete;
    virtual ~LatticeBase() = default;

    std::unique_ptr<Model> model;             ///<
    LatticeData data;                         ///<
    ZoneSettings zSet;                        ///<
    CudaStream_t kernelStream;                ///< CUDA Stream for kernel runs
    CudaStream_t inStream;                    ///< CUDA Stream for CPU->GPU momory copy
    CudaStream_t outStream;                   ///< CUDA Stream for GPU->CPU momory copy
    int reverse_save = 0;                     ///< Flag stating if recording (Now)
    int ZoneIter = 0;                         ///<
    setting_record_t settings_record;         ///< List of settings changes during the recording
    unsigned int settings_i;                  ///< Index in settings_record that is on the CUDA const
    int Record_Iter = 0;                      ///< Recorded iteration number (Now)
    int Iter = 0;                             ///< Iteration (Now) - "real" time of the simulation
    std::array<real_t, SETTINGS> settings{};  ///< Table of Settings (Now)
    std::array<double, GLOBALS> globals;      ///< Table of Globals
    std::function<int(int, int)> callback;    ///< Monitor callback
    int total_iterations = 0;                 ///< Total iteration number counter
    int segment_iterations = 0;               ///<
    int callback_iter = 1;                    ///<
    int num_snaps;                            ///<
    solidcontainer_t SC;                      ///<
    size_t particle_data_size_max = 0;        ///<
    rfi_t RFI;                                ///<
    SyntheticTurbulence ST;                   ///<
    std::string snapFileName;

   protected:
    static constexpr int maxSnaps = 33;

    const UnitEnv* units;
    std::unique_ptr<int[]> iSnaps = std::make_unique<int[]>(maxSnaps);  ///< Snapshot number (Now)
    int Snap, aSnap = 0;                                                ///< Snapshot and Adjoint Snapshot number (Now)

    virtual void initLatticeDerived() = 0;  /// Init derived lattice object when requested by handlers
    void MarkIteration();
    void FinalIteration();
    void InitialIteration(int segiter);
    int getSnap(int i) const;

   private:
    virtual int loadPrimal(const std::string& filename, int snap_ind) = 0;
    virtual void savePrimal(const std::string& filename, int snap_ind) const = 0;
#ifdef ADJOINT
    virtual int loadAdj(const std::string& filename, int asnap_ind) = 0;
    virtual void saveAdj(const std::string& filename, int asnap_ind) const = 0;
#endif
    virtual void clearAdjoint() = 0;

   public:
    virtual size_t getLocalSize() const = 0;
    virtual size_t getGlobalSize() const = 0;
    void initLattice();  /// Called by handlers

    template <class F>
    void setCallback(F&& fun) {
        callback = std::forward<F>(fun);
    }

    void setSetting(int i, real_t tmp);
    real_t getSetting(int i);
    void SetSetting(const Model::Setting& set, real_t val);
    void push_setting(int, real_t, real_t);  ///< Set the setting (and push to settings_record if recording)
    void pop_settings();                     ///< Pop the setting from settings_record


    virtual std::vector<int> shape() const = 0;
    virtual std::vector<real_t> getQuantity(const Model::Quantity& q, real_t scale = 1) = 0;
    virtual std::vector<big_flag_t> getFlags() const = 0;
    virtual std::vector<real_t> getField(const Model::Field& f) = 0;
    virtual std::vector<real_t> getFieldAdj(const Model::Field& f) = 0;
    virtual std::vector<real_t> getCoord(const Model::Coord& q, real_t scale = 1) = 0;

    virtual void setFlags(const std::vector<big_flag_t>& x) = 0;
    virtual void setField(const Model::Field& f, const std::vector<real_t>& x) = 0;
    virtual void setFieldAdjZero(const Model::Field& f) = 0;

    void CopyInParticles();
    void CopyOutParticles();

    void GenerateST();

    void getGlobals(real_t* tab);
    void calcGlobals();
    void clearGlobals();
    void clearGlobals_Adj();
    double getObjective();

    void startRecord();
    void rewindRecord();
    void stopRecord();

    void loadSolution(const std::string& filename);
    std::string saveSolution(const std::string& filename) const;

    void Iterate(int num_iters, int iter_type);
    void IterateTill(int it, int iter_type);
    void IterateAction(int action, int niter, int iter_type);
    void Iterate() { IterateT(ITER_NORM); }
    void IterateG() { IterateT(ITER_GLOBS); }
    void Stream() { IterateT(ITER_STREAM); }
    void IterateT(int iter_type) { Iterate(1, iter_type); }

    virtual void IterationPrimal(int, int, int) = 0;
    virtual void IterationAdjoint(int, int, int, int, int) = 0;
    virtual void IterationOptimization(int, int, int, int, int) = 0;
    virtual void RunAction(int, int, int, int) = 0;
    void RunAction(int action, int iter_type) { RunAction(action, Snap, (Snap + 1) % 2, iter_type); }

    virtual int EventLoop() { return 0; }  // Event loop does nothing by default
};

#endif  // LATTICEBASE_HPP
