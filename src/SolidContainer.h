#ifndef SOLIDCONTAINER_H
#define SOLIDCONTAINER_H

#include "pinned_allocator.hpp"
#include "RemoteForceInterface.h"
#include "SolidTree.h"

typedef rfi::RemoteForceInterface< rfi::ForceCalculator, rfi::RotParticle, rfi::ArrayOfStructures, real_t, pinned_allocator<real_t> > rfi_t;
typedef SolidTree< rfi_t > solidcontainer_t;

#endif // SOLIDCONTAINER_H
