#ifndef SOLIDCONTAINER_H
#define SOLIDCONTAINER_H

#include "pinned_allocator.hpp"
#include "RemoteForceInterface.h"
#include "SolidTree.h"
#include "SolidGrid.h"
#include "SolidAll.h"

typedef rfi::RemoteForceInterface< rfi::ForceCalculator, rfi::RotParticle, rfi::ArrayOfStructures, real_t, pinned_allocator<real_t> > rfi_t;

#if SOLID_CONTAINER == 1
    typedef SolidAll< rfi_t > solidcontainer_t;
#elif SOLID_CONTAINER == 2
    typedef SolidTree< rfi_t > solidcontainer_t;
#elif SOLID_CONTAINER == 3
    typedef SolidGrid< rfi_t > solidcontainer_t;
#else
    #error unknown value of SOLID_CONTAINER
#endif

#endif // SOLIDCONTAINER_H
