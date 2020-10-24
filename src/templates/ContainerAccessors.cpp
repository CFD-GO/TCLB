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

	// run functions go here
};

// Accessor for CartesianLatticeContainer
struct CartesianLatticeContainerAccessor {
	typedef CartesianLatticeContainerAccessor Accessor;
	typedef LatticeContainer Container;
	typedef LatticeContainer::index index;

	CudaDeviceFunction static LatticeContainer& container() {
		return constContainer;
	}

	// run functions go here

};

#define CONTAINERACCESSORS 1
#endif