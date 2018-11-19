ADJOINT=0
TEST=FALSE
OPT="(GF+RT+Outflow+GuoCM+PF_FOI)*autosym"

# GF: Guo Forcing;
#	This is using a higher order Forcing scheme
#	from the work of Guo et al. (2002) for the hydrodynamics
# RT: Ren Temporal
#	This is using the Temporal term included in the 
#	phase field equilibrium distribution function by
#	Ren et al. (2016)
# Outflow: 
# 	This is used for outflow boundaries, it is made as an
# 	option as it requires additional fields for calculations
# 	so results in a slower code.
# autosym:
# 	Allows symmetry node type flags introduced in v6.2
# Guo_CM:
#   Use transformation of the Guo's scheme in central moments space. 
#   Default is the He's scheme.
# PF_FOI: 
#   First order integration (rectangular disretization)
#   Deafult is second order integration (trapezoidal discretization)
