
AddDensity( name="f[0]", dx= 0, dy= 0, group="f")
AddDensity( name="f[1]", dx= 1, dy= 0, group="f")
AddDensity( name="f[2]", dx= 0, dy= 1, group="f")
AddDensity( name="f[3]", dx=-1, dy= 0, group="f")
AddDensity( name="f[4]", dx= 0, dy=-1, group="f")
AddDensity( name="f[5]", dx= 1, dy= 1, group="f")
AddDensity( name="f[6]", dx=-1, dy= 1, group="f")
AddDensity( name="f[7]", dx=-1, dy=-1, group="f")
AddDensity( name="f[8]", dx= 1, dy=-1, group="f")

AddQuantity(name="Rho",unit="kg/m3")
AddQuantity(name="T",unit="K")
AddQuantity(name="U",unit="m/s",vector=T)

AddDensity( name="T[0]", dx= 0, dy= 0, group="T")
AddDensity( name="T[1]", dx= 1, dy= 0, group="T")
AddDensity( name="T[2]", dx= 0, dy= 1, group="T")
AddDensity( name="T[3]", dx=-1, dy= 0, group="T")
AddDensity( name="T[4]", dx= 0, dy=-1, group="T")
AddDensity( name="T[5]", dx= 1, dy= 1, group="T")
AddDensity( name="T[6]", dx=-1, dy= 1, group="T")
AddDensity( name="T[7]", dx=-1, dy=-1, group="T")
AddDensity( name="T[8]", dx= 1, dy=-1, group="T")



AddSetting(name="omega", comment='one over relaxation time')
AddSetting(name="nu", omega='1.0/(3*nu + 0.5)', default=0.16666666, comment='viscosity')
AddSetting(name="InletVelocity", default="0m/s", comment='inlet velocity')
AddSetting(name="InletPressure", InletDensity='1.0+InletPressure/3', default="0Pa", comment='inlet pressure')
AddSetting(name="InletDensity", default=1, comment='inlet density')
AddSetting(name="InletTemperature", default=1, comment='inlet density')
AddSetting(name="InitTemperature", default=1, comment='inlet density')
AddSetting(name="FluidAlfa", default=1, comment='inlet density')

AddGlobal(name="OutFlux");

AddNodeType("Heater","ADDITIONALS")