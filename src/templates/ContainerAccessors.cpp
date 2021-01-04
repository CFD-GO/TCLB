#include "LatticeContainer.h"
#include "ArbitraryLatticeContainer.h"
#include "../cross.h"


#ifndef CONTAINERACCESSORS

// Accessor for ArbitraryLatticeContainer
struct ArbitraryLatticeContainerAccessor {
	typedef ArbitraryLatticeContainerAccessor Accessor;
	typedef ArbitraryLatticeContainer Container;
	typedef ArbitraryLatticeContainer::index index;

	CudaDeviceFunction static ArbitraryLatticeContainer& container() {
		return constArbitraryContainer;
	}

	/// Main function called by the Kernel for cartesian lattice
	/**
	 iterates over all elements and runs them with RunElement function.
	constContainer.dx/dy is to calculate only internal nodes
	*/
	template <class N> CudaDeviceFunction static void RunKernel() {
		N now;

		now.idx_.id = CudaThread.x + (CudaThread.y * CudaNumberOfThreads.x) + 
						(CudaBlock.x * CudaNumberOfThreads.x * CudaNumberOfThreads.y) + 
						(CudaBlock.y * CudaNumberOfThreads.x * CudaNumberOfThreads.y * constArbitraryContainer.bnx);
		now.Pre();
		if(now.idx_.id < constArbitraryContainer.latticeSize) {
			now.RunElement();
		} else {
			now.OutOfDomain();
		}
		now.Glob();
	}
};

// Accessor for CartesianLatticeContainer
struct CartesianLatticeContainerAccessor {
	typedef CartesianLatticeContainerAccessor Accessor;
	typedef LatticeContainer Container;
	typedef LatticeContainer::index index;

	CudaDeviceFunction static LatticeContainer& container() {
		return constContainer;
	}

	/// Main function called by the Kernel for arbitrary lattice
	/**
	 iterates over all elements and runs them with RunElement function.
	constContainer.dx/dy is to calculate only internal nodes
	*/
	template <class N> CudaDeviceFunction static void RunKernel() {
		N now;
		now.idx_.x = CudaThread.x + CudaBlock.z*CudaNumberOfThreads.x + constContainer.dx;
		now.idx_.y = CudaThread.y + CudaBlock.x*CudaNumberOfThreads.y + constContainer.dy;
		now.idx_.z = CudaBlock.y + constContainer.dz;

		#ifndef GRID3D
			for (; now.idx_.x < constContainer.nx; now.idx_.x += CudaNumberOfThreads.x) {
		#endif
			now.Pre();
			if (now.idx_.y < constContainer.fy) {
				now.RunElement();
			} else {
				now.OutOfDomain();
			}
			now.Glob();
		#ifndef GRID3D
			}
		#endif
	}
};

#define CONTAINERACCESSORS 1
#endif