
#Based on T.Lee: Eliminating parasitic currents in the lattice Boltzmann equation method for nonideal gases



AddDensity( name="f0", dx= 0, dy= 0, group="f")
AddDensity( name="f1", dx= 1, dy= 0, group="f")
AddDensity( name="f2", dx= 0, dy= 1, group="f")
AddDensity( name="f3", dx=-1, dy= 0, group="f")
AddDensity( name="f4", dx= 0, dy=-1, group="f")
AddDensity( name="f5", dx= 1, dy= 1, group="f")
AddDensity( name="f6", dx=-1, dy= 1, group="f")
AddDensity( name="f7", dx=-1, dy=-1, group="f")
AddDensity( name="f8", dx= 1, dy=-1, group="f")

AddField("rho",stencil2d=2);
AddField("nu",stencil2d=2);

AddStage("BaseIteration", "Run", save=Fields$group == "f", load=DensityAll$group == "f")

AddStage("CalcRho", save="rho", load=DensityAll$group == "f")
AddStage("CalcNu", save="nu", load=FALSE)
AddStage("InitRho", save="rho", load=FALSE)
AddStage("InitF",  save=Fields$group == "f", load=FALSE)
AddStage("InitF2", save=Fields$group == "f", load=FALSE)

AddAction("Iteration", c("BaseIteration","CalcRho","CalcNu"))
#AddAction("Init", c("InitRho","CalcNu","InitF"))
AddAction("Init", c("InitF2", "CalcRho","CalcNu"))

AddQuantity(name="Rho", unit="kg/m3");
AddQuantity(name="U", unit="m/s", vector=T);
AddQuantity(name="Nu", unit="kg/m3");
AddQuantity(name="P", unit="Pa");


AddSetting(name="omega", comment='one over relaxation time')
AddSetting(name="nu", omega='1.0/(3*nu + 0.5)', default=0.16666666, comment='viscosity')
AddSetting(name="InletVelocity", default="0m/s", comment='inlet velocity', zonal=TRUE)
AddSetting(name="InletPressure", InletDensity='1.0+InletPressure/3', default="0Pa", comment='inlet pressure', zonal=TRUE)
AddSetting(name="InletDensity", default=1, comment='inlet density', zonal=TRUE)
AddSetting(name="OutletDensity", default=1, comment='inlet density', zonal=TRUE)
AddSetting(name="InitDensity", comment='inlet density', zonal=TRUE)
AddSetting(name="WallDensity", comment='vapor/liquid density of wall', zonal=TRUE)

AddSetting(name="GravitationY", comment='Gravitation in the direction of y')
AddSetting(name="GravitationX", comment='Gravitation in the direction of x')
AddSetting(name="MovingWallVelocity", comment='Velocity of the MovingWall', zonal=TRUE)
AddSetting(name="WetDensity", comment='wet density', zonal=TRUE)
AddSetting(name="DryDensity", comment='dry density', zonal=TRUE)
AddSetting(name="Wetting", comment='wetting factor', zonal=TRUE)

AddSetting(name="LiquidDensity", comment="Density of liquid phase")
AddSetting(name="VaporDensity", comment="Density of vapor phase")
AddSetting(name="Beta", comment="Beta of Lee model")
AddSetting(name="Kappa", comment="Capilarity")

AddGlobal(name="MomentumX",        comment='momentum')
AddGlobal(name="MomentumY",        comment='momentum')
AddGlobal(name="Mass",         comment='mass')


AddNodeType(name="MovingWall", group="BOUNDARY")
AddNodeType(name="ForcedMovingWall", group="BOUNDARY")
AddNodeType(name="Wet", group="ADDITIONALS")
AddNodeType(name="Dry", group="ADDITIONALS")
