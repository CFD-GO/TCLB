
AddDensity( name="f[0]", dx= 0, dy= 0)
AddDensity( name="f[1]", dx= 1, dy= 0)
AddDensity( name="f[2]", dx= 0, dy= 1)
AddDensity( name="f[3]", dx=-1, dy= 0)
AddDensity( name="f[4]", dx= 0, dy=-1)
AddDensity( name="f[5]", dx= 1, dy= 1)
AddDensity( name="f[6]", dx=-1, dy= 1)
AddDensity( name="f[7]", dx=-1, dy=-1)
AddDensity( name="f[8]", dx= 1, dy=-1)

AddQuantity(name="Rho",unit="kg/m3")
AddQuantity(name="T",unit="K")
AddQuantity(name="U",unit="m/s",vector=T)

f = PV(DensityAll$name)

U = as.matrix(DensityAll[,c("dx","dy")])

AddDensity(
	name = paste("T[",0:8,"]"),
	dx   = c( 0, 1, 0,-1, 0, 1,-1,-1, 1),
	dy   = c( 0, 0, 1, 0,-1, 1, 1,-1,-1),
	dz   = c( 0, 0, 0, 0, 0, 0, 0, 0, 0),
	comment=paste("density T",0:8)
)

AddSetting(name="omega", comment='one over relaxation time')
AddSetting(name="nu", omega='1.0/(3*nu + 0.5)', default=1.6666666, comment='viscosity')
AddSetting(name="InletVelocity", default="0m/s", comment='inlet velocity')
AddSetting(name="InletPressure", InletDensity='1.0+InletPressure/3', default="0Pa", comment='inlet pressure')
AddSetting(name="InletDensity", default=1, comment='inlet density')
AddSetting(name="InletTemperature", default=1, comment='inlet density')
AddSetting(name="InitTemperature", default=1, comment='inlet density')
AddSetting(name="FluidAlfa", default=1, comment='inlet density')


