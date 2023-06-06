#ifndef SOLIDCONTAINER_H
#define SOLIDCONTAINER_H

#include "pinned_allocator.hpp"
#include "RemoteForceInterface.h"
const int max_cache_size = 16;
#include "SolidTree.h"
#include "SolidGrid.h"
#include "SolidAll.h"

typedef rfi::RemoteForceInterface< rfi::ForceCalculator, rfi::RotParticle, rfi::ArrayOfStructures, real_t, pinned_allocator<real_t> > rfi_t;
//typedef SolidTree< rfi_t > solidcontainer_t;
//typedef SolidAll< rfi_t > solidcontainer_t;
typedef SolidGrid< rfi_t > solidcontainer_t;

#endif // SOLIDCONTAINER_H
