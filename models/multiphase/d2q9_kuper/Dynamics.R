
AddDensity( name="f0", dx= 0, dy= 0, group="f")
AddDensity( name="f1", dx= 1, dy= 0, group="f")
AddDensity( name="f2", dx= 0, dy= 1, group="f")
AddDensity( name="f3", dx=-1, dy= 0, group="f")
AddDensity( name="f4", dx= 0, dy=-1, group="f")
AddDensity( name="f5", dx= 1, dy= 1, group="f")
AddDensity( name="f6", dx=-1, dy= 1, group="f")
AddDensity( name="f7", dx=-1, dy=-1, group="f")
AddDensity( name="f8", dx= 1, dy=-1, group="f")

AddField("phi",stencil2d=1);

AddStage("BaseIteration", "Run", save=Fields$group == "f", load=DensityAll$group == "f")
AddStage("CalcPhi", save="phi",load=DensityAll$group == "f")
AddStage("BaseInit", "Init", save=Fields$group == "f", load=DensityAll$group == "f")

AddAction("Iteration", c("BaseIteration","CalcPhi"))
AddAction("Init", c("BaseInit","CalcPhi"))

AddQuantity(name="Rho", unit="kg/m3");
AddQuantity(name="U", unit="m/s", vector=T);
AddQuantity(name="P", unit="Pa");
AddQuantity(name="F", unit="N", vector=T);

AddSetting(name="omega", comment='one over relaxation time')
AddSetting(name="nu", omega='1.0/(3*nu + 0.5)', default=0.16666666, comment='viscosity')
AddSetting(name="Velocity", default="0m/s", comment='inlet velocity')
AddSetting(name="Temperature", comment='temperature of the liquid/gas')
AddSetting(name="FAcc", comment='Multiplier of potential')

AddSetting(name="Magic", comment='K', default="0.01")
AddSetting(name="MagicA", comment='A in force calculation', default="-0.152")
AddSetting(name="MagicF", comment='Force multiplier', default="-0.66666666666666")

AddSetting(name="GravitationY", comment='Gravitation in the direction of y')
AddSetting(name="GravitationX", comment='Gravitation in the direction of x')
AddSetting(name="MovingWallVelocity", comment='Velocity of the MovingWall')
AddSetting(name="Density", comment='zonal density', zonal=TRUE)
AddSetting(name="Wetting", comment='wetting factor')

AddSetting(name="S0", default="0", comment='MRT Sx')
AddSetting(name="S1", default="0",comment='MRT Sx')
AddSetting(name="S2", default="0",comment='MRT Sx')
AddSetting(name="S3", default="-.333333333", comment='MRT Sx')
AddSetting(name="S4", default="0", comment='MRT Sx')
AddSetting(name="S5", default="0", comment='MRT Sx')
AddSetting(name="S6", default="0", comment='MRT Sx')
AddSetting(name="S7", default="1.-omega", comment='MRT Sx')
AddSetting(name="S8", default="1.-omega",  comment='MRT Sx')






#AddGlobal(name="MovingWallForceX", comment='force x')
#AddGlobal(name="MovingWallForceY", comment='force y')


AddGlobal(name="Pressure1",        comment='pressure at Obj1')
AddGlobal(name="Pressure2",        comment='pressure at Obj2')
AddGlobal(name="Pressure3",        comment='pressure at Obj3')
AddGlobal(name="Density1",         comment='density at Obj1')
AddGlobal(name="Density2",         comment='density at Obj2')
AddGlobal(name="Density3",         comment='density at Obj3')

AddGlobal(name="SumUsqr",         comment='Sumo o U**2')
AddGlobal(name="WallForceX", comment='force x')
AddGlobal(name="WallForceY", comment='force y')


AddNodeType(name="NMovingWall", group="BOUNDARY")
AddNodeType(name="MovingWall", group="BOUNDARY")
AddNodeType(name="ESymmetry",group="BOUNDARY")
AddNodeType(name="NSymmetry",group="BOUNDARY")
AddNodeType(name="SSymmetry",group="BOUNDARY")
