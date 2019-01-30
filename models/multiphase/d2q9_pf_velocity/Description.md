The 'd2q9_pf_velocity' model is a multiphase 2D(planar) lattice Boltzmann model for the simulation of immiscible fluids (at both high and low density ratios).

The base implementation uses a velocity based LBM for capturing the hydrodynamics of the flow and solves the conservative phase field equation for the interfacial dynamics. To enhance stability, a Multiple-Relaxation-Time collision operator is used.

The model currently has 4 options at compile time: 
 GF: Guo Forcing;
       This is using a higher order Forcing scheme
       from the work of Guo et al. (2002) for the hydrodynamics
 RT: Ren Temporal
       This is using the Temporal term included in the 
       phase field equilibrium distribution function by
       Ren et al. (2016)
 Outflow: 
       This is used for outflow boundaries, it is made as an
       option as it requires additional fields for calculations
       so results in a slower code.
 autosym:
       Allows symmetry node type flags introduced in v6.2

Publications:
 "    Improved locality of the phase-field lattice Boltzmann
      model for immiscible fluids at high density ratios          "
 Authors: A. Fakhari, T. Mitchell, C. Leonardi, D. Bolster (2017) 
 DOI: 10.1103/PhysRevE.96.053301

