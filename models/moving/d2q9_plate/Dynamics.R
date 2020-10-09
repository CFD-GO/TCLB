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

# Boundary initialization

AddDensity( name="avg_ux", group="avg_u")
AddDensity( name="avg_uy", group="avg_u")
AddDensity( name="avg_fx", group="avg_f")
AddDensity( name="avg_fy", group="avg_f")

AddNodeType(name="NVelocity", group="BOUNDARY")
AddNodeType(name="SPressure", group="BOUNDARY")

# Quantities - table of fields that can be exported from the LB lattice (like density, velocity etc)
#  name - name of the field
#  type - C type of the field, "real_t" - for single/double float, and "vector_t" for 3D vector single/double float
# Every field must correspond to a function in "Dynamics.c".
# If one have filed [something] with type [type], one have to define a function: 
# [type] get[something]() { return ...; }

AddQuantity(name="Rho",unit="kg/m3")
AddQuantity(name="U",unit="m/s",vector=T)
AddQuantity(name="U_AVG",unit="m/s",vector=T)
AddQuantity(name="F_AVG",unit="N/m3",vector=T)
AddQuantity(name="Solid",unit="1")

# Settings - table of settings (constants) that are taken from a .xml file
#  name - name of the constant variable
#  comment - additional comment
# You can state that another setting is 'derived' from this one stating for example: omega='1.0/(3*nu + 0.5)'

#AddSetting(name="omega", comment='one over relaxation time')
#AddSetting(name="nu", omega='1.0/(3*nu + 0.5)', default=0.16666666, comment='viscosity')
AddSetting(name="nu", default=0.16666666, comment='viscosity', zonal=T, unit="m2/s")
#AddSetting(name="Velocity", default=0, comment='inlet/outlet/init velocity', zonal=T, unit="m/s")
AddSetting(name="VelocityY", default=0, comment='inlet/outlet/init velocity', zonal=T, unit="m/s")
AddSetting(name="VelocityX", default=0, comment='inlet/outlet/init velocity', zonal=T, unit="m/s")
AddSetting(name="Density", default=1, comment='inlet/outlet/init density', zonal=T, unit="kg/m3")
AddSetting(name="Smag", default=1, comment='inlet density')

AddQuantity( name="RhoB",adjoint=T)
AddQuantity( name="UB",adjoint=T,vector=T)

# Globals - table of global integrals that can be monitored and optimized

AddGlobal(name="ForceX", comment='reaction force X', unit="N/m")
AddGlobal(name="ForceY", comment='reaction force Y', unit="N/m")
AddGlobal(name="Moment", comment='reaction force X', unit="N")
AddGlobal(name="PowerX", comment='reaction force Y', unit="W/m")
AddGlobal(name="PowerY", comment='reaction force X', unit="W/m")
AddGlobal(name="PowerR", comment='reaction force Y', unit="W/m")
AddGlobal(name="Power", comment='reaction force X', unit="W/m")
AddGlobal(name="Power2", comment='reaction force Y', unit="W/m")
AddGlobal(name="VolumeW", comment="Volume of moving body", unit="m2")


AddSetting(name="PDX", default=0, comment='plate diameter X', unit="m")
AddSetting(name="PDY", default=0, comment='plate diameter Y', unit="m")
AddSetting(name="PRAD", default=0, comment='cylinder radious', unit='m')
AddSetting(name="SM",   default=1, comment='smoothing diameter', unit="m")
AddSetting(name="SM_M", default=0, comment='smoothing bias')
AddSetting(name="EPSF", default=1, comment='boundary function, 0 - linear boundary,1 - third order boundary')
AddSetting(name="BF", default=0, comment='beta function bool')
AddSetting(name="PX", default=0, comment='plate position X', zonal=T, unit="m")
AddSetting(name="PY", default=0, comment='plate position Y', zonal=T, unit="m")
AddSetting(name="PR", default=0, comment='plate angle', zonal=T, unit="1")

AddObjective("EfficiencyX", PV("ForceX") * PV("Power") ^ (-1))
AddObjective("EfficiencyY", PV("ForceY") * PV("Power") ^ (-1))

AddSetting(name="ExternalForceX", default=0, comment='external force x', zonal=T, unit="N/m3")
AddSetting(name="ExternalForceY", default=0, comment='external force y', zonal=T, unit="N/m3")
AddNodeType(name="EPressure", group="BOUNDARY")
AddNodeType(name="EVelocity", group="BOUNDARY")
AddNodeType(name="Solid", group="BOUNDARY")
AddNodeType(name="Wall", group="BOUNDARY")
AddNodeType(name="WPressure", group="BOUNDARY")
AddNodeType(name="WVelocity", group="BOUNDARY")
AddNodeType(name="MRT", group="COLLISION")
