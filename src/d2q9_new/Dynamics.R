# Density - table of variables of LB Node to stream
#  name - variable name to stream
#  dx,dy,dz - direction of streaming
#  comment - additional comment

AddDensity( name="f[0]", dx= 0, dy= 0, group="f")
AddDensity( name="f[1]", dx= 1, dy= 0, group="f")
AddDensity( name="f[2]", dx= 0, dy= 1, group="f")
AddDensity( name="f[3]", dx=-1, dy= 0, group="f")
AddDensity( name="f[4]", dx= 0, dy=-1, group="f")
AddDensity( name="f[5]", dx= 1, dy= 1, group="f")
AddDensity( name="f[6]", dx=-1, dy= 1, group="f")
AddDensity( name="f[7]", dx=-1, dy=-1, group="f")
AddDensity( name="f[8]", dx= 1, dy=-1, group="f")

# Quantities - table of fields that can be exported from the LB lattice (like density, velocity etc)
#  name - name of the field
#  type - C type of the field, "real_t" - for single/double float, and "vector_t" for 3D vector single/double float
# Every field must correspond to a function in "Dynamics.c".
# If one have filed [something] with type [type], one have to define a function: 
# [type] get[something]() { return ...; }

AddQuantity(name="Rho",unit="kg/m3")
AddQuantity(name="U",unit="m/s",vector=T)

AddQuantity(name="A",unit="1",vector=T)


# Settings - table of settings (constants) that are taken from a .xml file
#  name - name of the constant variable
#  comment - additional comment
# You can state that another setting is 'derived' from this one stating for example: omega='1.0/(3*nu + 0.5)'

AddSetting(name="omega", comment='one over relaxation time')
AddSetting(name="nu", omega='1.0/(3*nu + 0.5)', default=1.6666666, comment='viscosity')
AddSetting(name="Velocity", default=0, comment='inlet/outlet/init velocity', zonal=T)
AddSetting(name="Pressure", default=0, comment='inlet/outlet/init density', zonal=T)
AddSetting(name="Smag", comment='Smagorinsky constant')

AddSetting(name="SL_U", comment='Shear Layer velocity')
AddSetting(name="SL_lambda", comment='Shear Layer lambda')
AddSetting(name="SL_delta", comment='Shear Layer disturbance')
AddSetting(name="SL_L", comment='Shear Layer length scale')


# Globals - table of global integrals that can be monitored and optimized

AddGlobal(name="PressureLoss", comment='pressure loss', unit="1mPa")
AddGlobal(name="OutletFlux", comment='pressure loss', unit="1m2/s")
AddGlobal(name="InletFlux", comment='pressure loss', unit="1m2/s")

AddNodeType("Smagorinsky", "LES")
AddNodeType("Stab", "ENTROPIC")
