ADJOINT=0
TEST=FALSE
OPT="(GF+RT+Outflow+GuoCM+PF_FOI+EDM_CMFOI+CMPCU)*autosym"

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
# PF_FOI: 
#   First order integration (rectangular disretization)
#   Deafult is second order integration (trapezoidal discretization)
# PCU_CM: 
# PCU use post collision velocity to back transform from CM
