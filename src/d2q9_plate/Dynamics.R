# Density - table of variables of LB Node to stream
#  name - variable name to stream
#  dx,dy,dz - direction of streaming
#  comment - additional comment

AddDensity( name="f0", dx= 0, dy= 0, group="f")
AddDensity( name="f1", dx= 1, dy= 0, group="f")
AddDensity( name="f2", dx= 0, dy= 1, group="f")
AddDensity( name="f3", dx=-1, dy= 0, group="f")
AddDensity( name="f4", dx= 0, dy=-1, group="f")
AddDensity( name="f5", dx= 1, dy= 1, group="f")
AddDensity( name="f6", dx=-1, dy= 1, group="f")
AddDensity( name="f7", dx=-1, dy=-1, group="f")
AddDensity( name="f8", dx= 1, dy=-1, group="f")

# Quantities - table of fields that can be exported from the LB lattice (like density, velocity etc)
#  name - name of the field
#  type - C type of the field, "real_t" - for single/double float, and "vector_t" for 3D vector single/double float
# Every field must correspond to a function in "Dynamics.c".
# If one have filed [something] with type [type], one have to define a function: 
# [type] get[something]() { return ...; }

AddQuantity(name="Rho",unit="kg/m3")
AddQuantity(name="U",unit="m/s",vector=T)
AddQuantity(name="Solid",unit="1")

# Settings - table of settings (constants) that are taken from a .xml file
#  name - name of the constant variable
#  comment - additional comment
# You can state that another setting is 'derived' from this one stating for example: omega='1.0/(3*nu + 0.5)'

#AddSetting(name="omega", comment='one over relaxation time')
#AddSetting(name="nu", omega='1.0/(3*nu + 0.5)', default=0.16666666, comment='viscosity')
AddSetting(name="tau0", comment='one over relaxation time')
AddSetting(name="nu", tau0='3*nu + 0.5', default=0.16666666, comment='viscosity')
AddSetting(name="Velocity", default=0, comment='inlet/outlet/init velocity', zonal=T)
AddSetting(name="Density", default=1, comment='inlet/outlet/init density', zonal=T)
AddSetting(name="Smag", default=1, comment='inlet density')

AddQuantity( name="RhoB",adjoint=T)
AddQuantity( name="UB",adjoint=T,vector=T)

# Globals - table of global integrals that can be monitored and optimized

AddGlobal(name="ForceX", comment='reaction force X')
AddGlobal(name="ForceY", comment='reaction force Y')
AddGlobal(name="Moment", comment='reaction force X')
AddGlobal(name="PowerX", comment='reaction force Y')
AddGlobal(name="PowerY", comment='reaction force X')
AddGlobal(name="PowerR", comment='reaction force Y')
AddGlobal(name="Power", comment='reaction force X')
AddGlobal(name="Power2", comment='reaction force Y')


AddSetting(name="PDX", default=0, comment='plate diameter X')
AddSetting(name="PDY", default=0, comment='plate diameter Y')
AddSetting(name="SM",   default=1, comment='smoothing diameter')
AddSetting(name="SM_M", default=0, comment='smoothing bias')
AddSetting(name="PX", default=0, comment='plate position X', zonal=T)
AddSetting(name="PY", default=0, comment='plate position Y', zonal=T)
AddSetting(name="PR", default=0, comment='plate angle', zonal=T)

