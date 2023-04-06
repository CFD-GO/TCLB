## Model for d2q9 SRT BKG-LBM
#     Density - performs streaming operation for us
#	

# Add the particle distribution functions as model Densities:
AddDensity( name="f_0", dx= 0, dy= 0)
AddDensity( name="f_1", dx= 1, dy= 0)
AddDensity( name="f_2", dx= 0, dy= 1)
AddDensity( name="f_3", dx=-1, dy= 0)
AddDensity( name="f_4", dx= 0, dy=-1)
AddDensity( name="f_5", dx= 1, dy= 1)
AddDensity( name="f_6", dx=-1, dy= 1)
AddDensity( name="f_7", dx=-1, dy=-1)
AddDensity( name="f_8", dx= 1, dy=-1)

# Add the quantities we wish to be exported
#    These quantities must be defined by a function in Dynamics.c
AddQuantity( name="U",unit="m/s", vector=TRUE )
AddQuantity( name="Rho",unit="kg/m3" )

# Add the settings which describes system constants defined in a .xml file
AddSetting( name="omega", comment='inverse of relaxation time')
AddSetting( name="nu", omega='1.0/(3*nu+0.5)', default=0.16666666, comment='viscosity')
AddSetting( name="VelocityX",default=0, comment='inlet/outlet/init velocity in x', zonal=TRUE )
AddSetting( name="VelocityY",default=0, comment='inlet/outlet/init velocity in y', zonal=TRUE )
AddSetting( name="GravitationX",default=0, comment='body/external acceleration', zonal=TRUE)
AddSetting( name="GravitationY",default=0, comment='body/external acceleration', zonal=TRUE)
AddSetting( name="Density",default=1, comment='Density')

AddNodeType(name="EPressure", group="BOUNDARY")
AddNodeType(name="WPressure", group="BOUNDARY")
AddNodeType(name="WVelocity", group="BOUNDARY")
AddNodeType(name="EVelocity", group="BOUNDARY")
AddNodeType(name="Solid", group="BOUNDARY")
AddNodeType(name="Wall", group="BOUNDARY")
AddNodeType(name="BGK", group="COLLISION")
