# Density - table of variables of LB Node to stream
#  name - variable name to stream
#  dx,dy,dz - direction of streaming
#  comment - additional comment

AddDensity( name="f[0]", dx= 0, dy= 0)
AddDensity( name="f[1]", dx= 1, dy= 0)
AddDensity( name="f[2]", dx= 0, dy= 1)
AddDensity( name="f[3]", dx=-1, dy= 0)
AddDensity( name="f[4]", dx= 0, dy=-1)
AddDensity( name="f[5]", dx= 1, dy= 1)
AddDensity( name="f[6]", dx=-1, dy= 1)
AddDensity( name="f[7]", dx=-1, dy=-1)
AddDensity( name="f[8]", dx= 1, dy=-1)

# Quantities - table of fields that can be exported from the LB lattice (like density, velocity etc)
#  name - name of the field
#  type - C type of the field, "real_t" - for single/double float, and "vector_t" for 3D vector single/double float
# Every field must correspond to a function in "Dynamics.c".
# If one have filed [something] with type [type], one have to define a function: 
# [type] get[something]() { return ...; }

AddQuantity(name="Rho",unit="kg/m3")
AddQuantity(name="U",unit="m/s",vector=T)

# Settings - table of settings (constants) that are taken from a .xml file
#  name - name of the constant variable
#  comment - additional comment
# You can state that another setting is 'derived' from this one stating for example: omega='1.0/(3*nu + 0.5)'

AddSetting(name="omega", comment='one over relaxation time')
AddSetting(name="nu", omega='1.0/(3*nu + 0.5)', default=1.6666666, comment='viscosity')
AddSetting(name="Velocity", default=0, comment='inlet/outlet/init velocity', zonal=T)
AddSetting(name="Density", default=1, comment='inlet/outlet/init density', zonal=T)

# Globals - table of global integrals that can be monitored and optimized

AddGlobal(name="PressureLoss", comment='pressure loss')

