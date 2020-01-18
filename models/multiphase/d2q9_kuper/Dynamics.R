
AddDensity( name="f0", dx= 0, dy= 0, group="f")
AddDensity( name="f1", dx= 1, dy= 0, group="f")
AddDensity( name="f2", dx= 0, dy= 1, group="f")
AddDensity( name="f3", dx=-1, dy= 0, group="f")
AddDensity( name="f4", dx= 0, dy=-1, group="f")
AddDensity( name="f5", dx= 1, dy= 1, group="f")
AddDensity( name="f6", dx=-1, dy= 1, group="f")
AddDensity( name="f7", dx=-1, dy=-1, group="f")
AddDensity( name="f8", dx= 1, dy=-1, group="f")

if (Options$wallNormalBC) {
 
    AddField( name="nw_x", stencil2d=1, group="nw")
    AddField( name="nw_y", stencil2d=1, group="nw")
}

AddField("rho_n",stencil2d=1, group="rho_n");

AddStage("BaseIteration", "Run", save=Fields$group == "f"| Fields$group=="nw" , load=DensityAll$group == "f")
AddStage("CalcRhoSC", save=Fields$group == "rho_n",load=DensityAll$group == "f")
AddStage("BaseInit", "Init", save=Fields$group == "f" | Fields$group == "rho_n" , load=DensityAll$group == "f")

if (Options$wallNormalBC) {
    AddStage("CalcWallNormall", "CalcNormal",   
             save=Fields$group=="nw",
             fixedPoint=TRUE
             ) 
    AddAction("Init", c("BaseInit","CalcWallNormall"))
    AddQuantity(name="WallNormal", vector=T);

} else {
    AddAction("Init", c("BaseInit"))
}
AddAction("Iteration", c("BaseIteration","CalcRhoSC"))




AddQuantity(name="Rho", unit="kg/m3");
AddQuantity(name="U", unit="m/s", vector=T);
AddQuantity(name="P", unit="Pa");
AddQuantity(name="F", unit="N", vector=T);


AddQuantity(name="DEBUG", vector=T);

AddSetting(name="omega", comment='relaxation factor', default=1)
AddSetting(name="nu", omega='1.0/(3*nu + 0.5)', comment='viscosity')
AddSetting(name="Velocity", default="0m/s", comment='inlet velocity')
AddSetting(name="Temperature", comment='temperature of the liquid/gas')
AddSetting(name="FAcc", comment='Multiplier of potential')

AddSetting(name="LVRho_phi_dr", default=1, , zonal=TRUE, comment="(wa < 90 <-> LVRho_phi_dr>=1) |  (wa > 90 <-> LVRho_phi_dr>=0) Wetting toning parameter, see DOI: 10.1103/PhysRevE.100.053313")

AddSetting(name="LVRho_ulimit", default=0.01, , zonal=TRUE, comment="Upper limiting value of rho_w see DOI: 10.1103/PhysRevE.100.053313")
AddSetting(name="LVRho_llimit", default=3.2, , zonal=TRUE,  comment="Lower limiting value of rho_w see DOI: 10.1103/PhysRevE.100.053313")


AddSetting(name="WallSmoothingMagic", default=0.12, comment="Wall normal smoothing parameter, higher - more smoothed")

AddSetting(name="Magic", comment='K', default="0.01")
AddSetting(name="MagicA", comment='A in force calculation', default="-0.152")
AddSetting(name="MagicF", comment='Force multiplier', default="-0.66666666666666")

AddSetting(name="GravitationY", comment='Gravitation in the direction of y')
AddSetting(name="GravitationX", comment='Gravitation in the direction of x')
AddSetting(name="MovingWallVelocity", comment='Velocity of the MovingWall')
AddSetting(name="Density", comment='zonal density', zonal=TRUE)
AddSetting(name="Wetting", comment='wetting factor')


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
