#ifndef ARBLATTICELAUNCHER_H
#define ARBLATTICELAUNCHER_H

#include "ArbLatticeContainer.hpp"
#include "LatticeData.hpp"

struct ArbLatticeLauncher {
    ArbLatticeContainer container;

    template <eOperationType I, eCalculateGlobals G, eStage S>
    void RunBorder(CudaStream_t stream, const LatticeData& data) const;
    template <eOperationType I, eCalculateGlobals G, eStage S>
    void RunInterior(CudaStream_t stream, const LatticeData& data) const;
};

#endif  // ARBLATTICELAUNCHER_H
