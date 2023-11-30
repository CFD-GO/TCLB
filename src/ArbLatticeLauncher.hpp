#ifndef ARBLATTICELAUNCHER_HPP
#define ARBLATTICELAUNCHER_HPP

#include "ArbLatticeAccess.hpp"
#include "ArbLatticeLauncher.h"
#include "GetThreads.h"
#include "LatticeData.hpp"
#include "Node.hpp"

template <eOperationType I, eCalculateGlobals G, eStage S>
struct ArbLatticeExecutor : public LinearExecutor {
    ArbLatticeContainer container;
    LatticeData data;
    unsigned offset;  /// Starting offset for the iteration space, allows the reuse of the executor for both the border and interior

    CudaDeviceFunction void Execute() const {
        using LA = ArbLatticeAccess;
        using N = Node<LA, I, G, S>;
        const int i = threadID(CudaThread, CudaBlock, CudaNumberOfThreads);
        if (inRange(i)) {
            const unsigned node_lid = offset + i;
            ArbLatticeAccess acc(node_lid, container);
            N now(acc, data);
            now.RunElement();
        }
    }
};

template <eOperationType I, eCalculateGlobals G, eStage S>
void ArbLatticeLauncher::RunBorder(CudaStream_t stream, const LatticeData& data) const {
    const ArbLatticeExecutor<I, G, S> executor{{container.num_border_nodes}, container, data, 0};
    LaunchExecutorAsync(executor, stream);
}

template <eOperationType I, eCalculateGlobals G, eStage S>
void ArbLatticeLauncher::RunInterior(CudaStream_t stream, const LatticeData& data) const {
    const ArbLatticeExecutor<I, G, S> executor{{container.num_interior_nodes}, container, data, container.num_border_nodes};
    LaunchExecutorAsync(executor, stream);
}

#endif  // ARBLATTICELAUNCHER_HPP
