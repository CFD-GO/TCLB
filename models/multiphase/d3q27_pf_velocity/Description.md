The 'd3q27_pf_velocity' model is a multiphase 3D lattice Boltzmann model for the simulation of immiscible fluids (at both high and low density ratios).

The base implementation uses a velocity based LBM for capturing the hydrodynamics of the flow and solves the conservative phase field equation for the interfacial dynamics. To enhance stability, a Weighted-Multiple-Relaxation-Time collision operator is used.

The model currently has 3 options at compile time: 
	- OutFlow; this enables convective and neumann outflow conditions. It requires extra memory access and is thus added as a compile option (faster code without).
	- BGK; this is in existence for the sole reason of initial testing, however if you would like to use a BGK collision operator - this flag is necessary.
	- autosym; both of these options can be compiled with symmetry conditions. 
