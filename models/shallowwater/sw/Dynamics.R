AddDescription( "Shallow water equation", "

Lattice Boltzmann Method for Shallow Water equation on a D2Q9 lattice. 
Model has adjoint capabilities for unsteady optimization.

")


AddDensity( name="f0", dx= 0, dy= 0, group="f")
AddDensity( name="f1", dx= 1, dy= 0, group="f")
AddDensity( name="f2", dx= 0, dy= 1, group="f")
AddDensity( name="f3", dx=-1, dy= 0, group="f")
AddDensity( name="f4", dx= 0, dy=-1, group="f")
AddDensity( name="f5", dx= 1, dy= 1, group="f")
AddDensity( name="f6", dx=-1, dy= 1, group="f")
AddDensity( name="f7", dx=-1, dy=-1, group="f")
AddDensity( name="f8", dx= 1, dy=-1, group="f")
AddDensity( name="w", group="w", parameter=T)

AddQuantity( name="Rho",unit="m")
AddQuantity( name="U",unit="m/s",vector=T)
AddQuantity( name="RhoB",adjoint=T)
AddQuantity( name="UB",adjoint=T,vector=T)
AddQuantity( name="W")
AddQuantity( name="WB",adjoint=T)

AddSetting(name="omega", comment='one over relaxation time')
AddSetting(name="nu", omega='1.0/(3*nu + 0.5)', default=0.16666666, comment='viscosity')
AddSetting(name="InletVelocity", default="0m/s", comment='inlet velocity')
AddSetting(name="InletPressure", InletDensity='1.0+InletPressure/3', default="0Pa", comment='inlet pressure')
AddSetting(name="InletDensity", default=1, comment='inlet density')
AddSetting(name="Gravity", default=1, comment='inlet density')
AddSetting(name="SolidH", default=1, comment='inlet density')
AddSetting(name="EnergySink", default=0, comment='inlet density')

AddSetting(name="Height", default=0, zonal=T)

AddGlobal(name="PressDiff", comment='pressure loss')
AddGlobal(name="TotalDiff", comment='total variation of velocity')
AddGlobal(name="Material", comment='total material')
AddGlobal(name="EnergyGain", comment='pressure loss')

AddNodeType(name="Obj1",group="OBJECTIVE")
