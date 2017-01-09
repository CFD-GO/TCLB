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



##########################################################
### Phase field related
AddDensity( name="h[0]", dx= 0, dy= 0, group="h")
AddDensity( name="h[1]", dx= 1, dy= 0, group="h")
AddDensity( name="h[2]", dx= 0, dy= 1, group="h")
AddDensity( name="h[3]", dx=-1, dy= 0, group="h")
AddDensity( name="h[4]", dx= 0, dy=-1, group="h")
AddDensity( name="h[5]", dx= 1, dy= 1, group="h")
AddDensity( name="h[6]", dx=-1, dy= 1, group="h")
AddDensity( name="h[7]", dx=-1, dy=-1, group="h")
AddDensity( name="h[8]", dx= 1, dy=-1, group="h")

AddQuantity(name="Normal",unit="1/m",vector=T)
AddQuantity(name="PhaseField",unit="1")

AddSetting(name="IntWidth", default=0.33333, comment='Interface width')
AddSetting(name="Mobility", default=0.001, comment='Mobility')
AddSetting(name="PhaseField", 
           default=0.5, 
           comment='Phase Field marker scalar', 
           zonal=T
           )



AddSetting(name="OverwriteVelocityField", default="0")
#########################################################


# THIS QUANTITIES ARE NEEDED FOR PYTHON INTEGRATION EXAMPLE
# COMMENT OUT FOR PERFORMANCE
# If present thei are used:
# As VelocityX/Y for Boundary conditions
# As mass force (+ GravitationX/Y) in fluid
# If OverwriteVelocityField==1, this will be used to overwrite velocity
AddDensity( name="BC[0]", group="BC", parameter=TRUE)
AddDensity( name="BC[1]", group="BC", parameter=TRUE)





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
# You can state that another setting is 'derived' from this one stating for example: RelaxationRate='1.0/(3*nu + 0.5)'

AddSetting(
           name="RelaxationRate", 
           S2='1-RelaxationRate',       
           comment='one over relaxation time'
            )
AddSetting(name="Viscosity", RelaxationRate='1.0/(3*Viscosity + 0.5)', default=0.16666666, comment='viscosity')
AddSetting(name="VelocityX", default=0, comment='inlet/outlet/init velocity', zonal=T)
AddSetting(name="VelocityY", default=0, comment='inlet/outlet/init velocity', zonal=T)
AddSetting(name="Pressure", default=0, comment='inlet/outlet/init density', zonal=T)

AddSetting(name="GravitationX", default=0)
AddSetting(name="GravitationY", default=0)


AddSetting(name="S2", default="0", comment='MRT Sx')
AddSetting(name="S3", default="0", comment='MRT Sx')
AddSetting(name="S4", default="0", comment='MRT Sx')


# Globals - table of global integrals that can be monitored and optimized

AddGlobal(name="PressureLoss", comment='pressure loss', unit="1mPa")
AddGlobal(name="OutletFlux", comment='pressure loss', unit="1m2/s")
AddGlobal(name="InletFlux", comment='pressure loss', unit="1m2/s")


AddNodeType(name="NPressure",group="BOUNDARY")
AddNodeType(name="SPressure",group="BOUNDARY")
AddNodeType(name="WPressure",group="BOUNDARY")
AddNodeType(name="EPressure",group="BOUNDARY")


AddNodeType(name="WVelocity",group="BOUNDARY")
AddNodeType(name="EVelocity",group="BOUNDARY")
