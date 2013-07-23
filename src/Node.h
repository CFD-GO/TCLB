#ifndef NODE_H
#define NODE_H 1

#include "Node_types.h"

struct
//__align__(16)
Node {
  #ifndef ADJOINT
    #include "Dynamics.h"
  #else
    #include "Dynamics_b.h"
  #endif
  #include "Dynamics.hp"
  #ifdef ADJOINT
    #include "Dynamics_b.hp"
    #include "Dynamics_adj.hp"
  #endif
};
struct Node_Globs;

#endif
