
AddDensity(
	name = paste("f",0:8,sep=""),
	dx   = c( 0, 1, 0,-1, 0, 1,-1,-1, 1),
	dy   = c( 0, 0, 1, 0,-1, 1, 1,-1,-1),
	dz   = c( 0, 0, 0, 0, 0, 0, 0, 0, 0),
	comment = paste("streamed density F",0:8),
	group = "f"
)

U = as.matrix(Density[,c("dx","dy")])

AddDensity(
	name = paste("fs",0:8,sep=""),
	dx   = 0,
	dy   = 0,
	dz   = 0,
	comment = paste("density F",0:8),
	group = "fs"
)

AddDensity(
	name = paste("phi",0:8,sep=""),
	dx   = c( 0, 1, 0,-1, 0, 1,-1,-1, 1),
	dy   = c( 0, 0, 1, 0,-1, 1, 1,-1,-1),
	dz   = c( 0, 0, 0, 0, 0, 0, 0, 0, 0),
	comment = paste("density F",0:8),
	group = "phi"
)

AddDensity(
	name="w",
	parameter=T
);

AddQuantity(name="Rho", unit="kg/m3");
AddQuantity(name="U", unit="m/s", vector=T);
#AddQuantity(name="P", unit="Pa");
#AddQuantity(name="F", unit="N", vector=T);
AddQuantity(name="RhoB", adjoint=T);
AddQuantity(name="UB", adjoint=T, vector=T);
AddQuantity(name="WB", adjoint=T);
AddQuantity(name="W");

f  = PV(Density$name[Density$group=="f"])
fs = PV(Density$name[Density$group=="fs"])
ph = PV(Density$name[Density$group=="phi"])

AddSetting(name="omega", comment='one over relaxation time')
AddSetting(name="nu", omega='1.0/(3*nu + 0.5)', default=0.16666666, comment='viscosity')
AddSetting(name="InletVelocity", default="0m/s", comment='inlet velocity')
AddSetting(name="InletPressure", InletDensity='1.0+InletPressure/3', default="0Pa", comment='inlet pressure')
AddSetting(name="InletDensity", default=1, comment='inlet density')
AddSetting(name="OutletDensity", default=1, comment='inlet density')
AddSetting(name="InitDensity", comment='inlet density')
AddSetting(name="WallDensity", comment='vapor/liquid density of wall')
AddSetting(name="Temperature", comment='temperature of the liquid/gas')
AddSetting(name="FAcc", comment='Multiplier of potential')
AddSetting(name="Magic", comment='K')
AddSetting(name="MagicA", comment='A in force calculation')
AddSetting(name="MagicF", comment='Force multiplier')
AddSetting(name="GravitationY", comment='Gravitation in the direction of y')
AddSetting(name="GravitationX", comment='Gravitation in the direction of x')
AddSetting(name="MovingWallVelocity", comment='Velocity of the MovingWall')
AddSetting(name="WetDensity", comment='wet density')
AddSetting(name="DryDensity", comment='dry density')
AddSetting(name="Wetting", comment='wetting factor')

AddGlobal(name="MovingWallForceX", comment='force x')
AddGlobal(name="MovingWallForceY", comment='force y')
AddGlobal(name="Pressure1",        comment='pressure at Obj1')
AddGlobal(name="Pressure2",        comment='pressure at Obj2')
AddGlobal(name="Pressure3",        comment='pressure at Obj3')
AddGlobal(name="Density1",         comment='density at Obj1')
AddGlobal(name="Density2",         comment='density at Obj2')
AddGlobal(name="Density3",         comment='density at Obj3')
AddGlobal(name="FluidVelocityX", comment='velocity x')
