# Density - table of variables of LB Node to stream
#  name - variable name to stream
#  dx,dy,dz - direction of streaming
#  comment - additional comment

Density = table_from_text("
	name dx dy dz   comment
	f[0]  0  0  0   density0
	f[1]  1  0  0   density1
	f[2] -1  0  0   density2
")

# Quantities - table of fields that can be exported from the LB lattice (like density, velocity etc)
#  name - name of the field
#  type - C type of the field, "type_f" - for single/double float, and "type_v" for 3D vector single/double float
# Every field must correspond to a function in "Dynamics.c".
# If one have filed [something] with type [type], one have to define a function: 
# [type] get[something]() { return ...; }

Quantities = table_from_text("
	name   type
	Rho    type_f
	U      type_v
")

# Settings - table of settings (constants) that are taken from a .clb file
#  name - name of the constant variable
#  derived - name of a constant that is calculated from this variable
#  equation - equation for calculating the above variable
#  comment - additional comment
# If one have filled [something] with the "derived" property set to [variable], and "equation" to "1+[something]" it means
# that if [something] is set in the .clb file, then the [variable] will be filled with 1+[something]

Settings = table_from_text("
	name                 derived                equation   comment
	omega                     NA                      NA   'one over relaxation time'
	nu                     omega      '1.0/(3*nu + 0.5)'   'viscosity'
	InletVelocity             NA                      NA   'inlet velocity'
	InletPressure   InletDensity   '1.0+InletPressure/3'   'inlet pressure'
	InletDensity              NA                      NA   'inlet density'
")


Globals = table_from_text("
	name            in_objective   comment
	PressureLoss    1              'pressure loss'
")

#-----------------------------------------------------------------------------------
# Variables for Dynamics.c.Rt
#

f = PV(Density$name)
U = as.matrix(Density[,c("dx","dy")])
