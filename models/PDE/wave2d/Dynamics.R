AddDensity( name="h", dx= 0, dy= 0, group="f")
AddDensity( name="u", dx= 0, dy= 0, group="f")
AddDensity( name="h1", dx= 1, dy= 0, group="f")
AddDensity( name="h2", dx= 0, dy= 1, group="f")
AddDensity( name="h3", dx=-1, dy= 0, group="f")
AddDensity( name="h4", dx= 0, dy=-1, group="f")
AddDensity( name="w", group="w")

AddQuantity( name="H")
AddQuantity( name="W")
AddQuantity( name="WB",adjoint=T)
AddQuantity( name="HB",adjoint=T)

AddSetting(name="WaveK", comment='coeff')
AddSetting(name="SolidH", comment='H of solid')
AddSetting(name="Loss", comment='u multipiler')

AddGlobal(name="TotalDiff", comment='total diff')

