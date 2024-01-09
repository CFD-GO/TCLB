#ifndef ACARBITRARYLATTICE_H
#define ACARBITRARYLATTICE_H

#include "../CommonHandler.h"
#include "Action.h"
#include "GenericAction.h"
#include "GenericContainer.h"
#include "vHandler.h"

class acArbitraryLattice : public Action {
   public:
    static std::string xmlname;
    int Init();
};

#endif  // ACARBITRARYLATTICE_H
