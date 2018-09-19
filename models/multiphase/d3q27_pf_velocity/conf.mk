ADJOINT=0
TEST=FALSE
OPT="(OutFlow+BGK+RT)*autosym"
# SC: Solid Contact
# 	This option currently fixes the bottom layer of nodes to be 
# 	solid with the contact angle defined in input.
# RT: Ren Temporal Correction
#	Utilises the previous velocity and phase value at each cell
#	to ensure consistency of recovered Allen-Cahn eqn.
