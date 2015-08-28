
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
AddQuantity(name="C",unit="1")
AddQuantity(name="Ct",unit="1")
AddQuantity(name="Cl_eq",unit="1")
AddQuantity(name="Solid",unit="1")
AddQuantity(name="U",unit="m/s",vector=T)
AddQuantity(name="K",unit="1/m")
AddQuantity(name="Theta",unit="1")

AddDensity( name="g[0]", dx= 0, dy= 0, group="g")
AddDensity( name="g[1]", dx= 1, dy= 0, group="g")
AddDensity( name="g[2]", dx= 0, dy= 1, group="g")
AddDensity( name="g[3]", dx=-1, dy= 0, group="g")
AddDensity( name="g[4]", dx= 0, dy=-1, group="g")
AddDensity( name="g[5]", dx= 1, dy= 1, group="g")
AddDensity( name="g[6]", dx=-1, dy= 1, group="g")
AddDensity( name="g[7]", dx=-1, dy=-1, group="g")
AddDensity( name="g[8]", dx= 1, dy=-1, group="g")

AddDensity( name="h[0]", dx= 0, dy= 0, group="h")
AddDensity( name="h[1]", dx= 1, dy= 0, group="h")
AddDensity( name="h[2]", dx= 0, dy= 1, group="h")
AddDensity( name="h[3]", dx=-1, dy= 0, group="h")
AddDensity( name="h[4]", dx= 0, dy=-1, group="h")
AddDensity( name="h[5]", dx= 1, dy= 1, group="h")
AddDensity( name="h[6]", dx=-1, dy= 1, group="h")
AddDensity( name="h[7]", dx=-1, dy=-1, group="h")
AddDensity( name="h[8]", dx= 1, dy=-1, group="h")

AddField( name="fi_s", dx=c(-1,1),dy=c(-1,1), comment="solidification")
AddDensity( name="fi_s" )

AddDensity( name="Cs" )

AddSetting(name="nu", comment='viscosity', unit="m2/s")
AddSetting(name="FluidAlfa", default=1, comment='inlet density', unit="m2/s")
AddSetting(name="SoluteDiffusion", comment='Solute diffusion coefficient in liquid', unit="m2/s")
AddSetting(name="C0", comment='Concentration 0')
AddSetting(name="T0", comment='Temperature 0', unit="K")
AddSetting(name="Teq", comment='Equilibrium temperature at interface', unit="K")

AddSetting(name="Velocity", default="0m/s", comment='fluid velocity', zonal=TRUE, unit="m/s")
AddSetting(name="Pressure", comment='pressure', zonal=TRUE, unit="Pa")
AddSetting(name="Temperature", comment='temperature', zonal=TRUE, unit="K")
AddSetting(name="Concentration", comment='concentration', zonal=TRUE)
AddSetting(name="Theta0", comment='Angle of preferential growth', zonal=TRUE, unit="d")

AddSetting(name="PartitionCoef", comment='Partition coefficient k')
AddSetting(name="LiquidusSlope", comment='Liquidus slope m', unit="K")
AddSetting(name="GTCoef", comment='Gibbs-Thomson coefficient gamma',unit="mK")
AddSetting(name="SurfaceAnisotropy", comment='Degree of anisotropy of surface energy')
AddSetting(name="SoluteCapillar", comment='Solutal capillary length d_0', unit="m")
AddSetting(name="Buoyancy", comment="Buoyancy Boussinesq approximation", unit="m/s2K")


AddGlobal(name="OutFlux");
AddGlobal(name="Material");

AddNodeType("Heater","ADDITIONALS")
AddNodeType("ForceTemperature","ADDITIONALS")
AddNodeType("ForceConcentration","ADDITIONALS")
AddNodeType("Seed","ADDITIONALS")
AddNodeType("Obj","OBJECTIVE")

