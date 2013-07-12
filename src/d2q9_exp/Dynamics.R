# Density - table of variables of LB Node to stream
#  name - variable name to stream
#  dx,dy,dz - direction of streaming
#  comment - additional comment

Density = table_from_text("
	name dx dy dz   comment
	f[0]  0  0  0   density0
	f[1]  1  0  0   density1
	f[2]  0  1  0   density2
	f[3] -1  0  0   density3
	f[4]  0 -1  0   density4
	f[5]  1  1  0   density5
	f[6] -1  1  0   density6
	f[7] -1 -1  0   density7
	f[8]  1 -1  0   density8
")

# Quantities - table of fields that can be exported from the LB lattice (like density, velocity etc)
#  name - name of the field
#  type - C type of the field, "real_t" - for single/double float, and "vector_t" for 3D vector single/double float
# Every field must correspond to a function in "Dynamics.c".
# If one have filed [something] with type [type], one have to define a function: 
# [type] get[something]() { return ...; }

Quantities = table_from_text("
	name   type
	Rho    real_t
	U      vector_t
")

# Settings - table of settings (constants) that are taken from a .clb file
#  name - name of the constant variable
#  derived - name of a constant that is calculated from this variable
#  equation - equation for calculating the above variable
#  comment - additional comment
# If one have filled [something] with the "derived" property set to [variable], and "equation" to "1+[something]" it means
# that if [something] is set in the .clb file, then the [variable] will be filled with 1+[something]

Settings = table_from_text("
	name   derived             equation   comment
	omega       NA                   NA   'one over relaxation time'
	nu       omega   '1.0/(3*nu + 0.5)'   'viscosity'
	UX          NA                   NA   'inlet velocity'
")
