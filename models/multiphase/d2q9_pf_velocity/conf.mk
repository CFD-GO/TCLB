ADJOINT=0
TEST=FALSE
OPT="(GF+RT+Outflow+debug+CM)*autosym"

# GF: Guo Forcing - feature in MRT model;
#	This is using a higher order Forcing scheme
#	from the work of Guo et al. (2002) for the hydrodynamics
# RT: Ren Temporal - feature in MRT model
#	This is using the Temporal term included in the 
#	phase field equilibrium distribution function by
#	Ren et al. (2016)
# debug: Enables tracking of momentum and force globals 
# CM: Applies the Central moments relaxation

# Boundary Conditions:
# Outflow: 
# 	This is used for outflow boundaries, it is made as an
# 	option as it requires additional fields for calculations
# 	so results in a slower code.
# autosym:
# 	Allows symmetry node type flags introduced in v6.2
