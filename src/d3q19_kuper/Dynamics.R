MRTMAT = matrix(c(
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
-30,-11,-11,-11,-11,-11,-11,8,8,8,8,8,8,8,8,8,8,8,8,
12,-4,-4,-4,-4,-4,-4,1,1,1,1,1,1,1,1,1,1,1,1,
0,1,-1,0,0,0,0,1,-1,1,-1,1,-1,1,-1,0,0,0,0,
0,-4,4,0,0,0,0,1,-1,1,-1,1,-1,1,-1,0,0,0,0,
0,0,0,1,-1,0,0,1,1,-1,-1,0,0,0,0,1,-1,1,-1,
0,0,0,-4,4,0,0,1,1,-1,-1,0,0,0,0,1,-1,1,-1,
0,0,0,0,0,1,-1,0,0,0,0,1,1,-1,-1,1,1,-1,-1,
0,0,0,0,0,-4,4,0,0,0,0,1,1,-1,-1,1,1,-1,-1,
 0,2,2,-1,-1,-1,-1,1,1,1,1,1,1,1,1,-2,-2,-2,-2,
0,-4,-4,2,2,2,2,1,1,1,1,1,1,1,1,-2,-2,-2,-2,
 0,0,0,1,1,-1,-1,1,1,1,1,-1,-1,-1,-1,0,0,0,0,
0,0,0,-2,-2,2,2,1,1,1,1,-1,-1,-1,-1,0,0,0,0,
 0,0,0,0,0,0,0,1,-1,-1,1,0,0,0,0,0,0,0,0,
 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,-1,-1,1,
 0,0,0,0,0,0,0,0,0,0,0,1,-1,-1,1,0,0,0,0,
0,0,0,0,0,0,0,1,-1,1,-1,-1,1,-1,1,0,0,0,0,
0,0,0,0,0,0,0,-1,-1,1,1,0,0,0,0,1,-1,1,-1,
0,0,0,0,0,0,0,0,0,0,0,1,1,-1,-1,-1,-1,1,1
),19,19)

v = diag(t(MRTMAT) %*% MRTMAT)
MRTMAT.inv = diag(1/v) %*% t(MRTMAT)

selU = c(4,6,8)

AddDensity(
	name = paste("f",0:18,sep=""),
	dx   = MRTMAT[,selU[1]],
	dy   = MRTMAT[,selU[2]],
	dz   = MRTMAT[,selU[3]],
	comment=paste("density F",0:18),
	group="f"
)

AddField("phi",stencil3d=1);

AddStage("BaseIteration", "Run", save=Fields$group == "f", load=DensityAll$group == "f")
AddStage("CalcPhi", save="phi",load=DensityAll$group == "f")
AddStage("BaseInit", "Init", save=Fields$group == "f", load=DensityAll$group == "f")

AddAction("Iteration", c("BaseIteration","CalcPhi"))
AddAction("Init", c("BaseInit","CalcPhi"))

AddQuantity(name="Rho", unit="kg/m3");
AddQuantity(name="U", unit="m/s", vector=T);
AddQuantity(name="P", unit="Pa");
AddQuantity(name="Phi", unit="1");
AddQuantity(name="F", unit="N", vector=T);

AddSetting(name="omega", comment='one over relaxation time')
AddSetting(name="nu", omega='1.0/(3*nu + 0.5)', default=0.16666666, comment='viscosity')
AddSetting(name="InletVelocity", default="0m/s", comment='inlet velocity')
AddSetting(name="Temperature", comment='temperature of the liquid/gas')
AddSetting(name="FAcc", comment='Multiplier of potential', default="1")

AddSetting(name="BoundaryVelocity_x", default="0m/s", comment='boundary velocity')
AddSetting(name="BoundaryVelocity_y", default="0m/s", comment='boundary velocity')
AddSetting(name="BoundaryVelocity_z", default="0m/s", comment='boundary velocity')
AddSetting(name="Boundary_rho", default="0m/s", comment='boundary velocity')

AddSetting(name="Magic", comment='K', default="0.01")
AddSetting(name="MagicA", comment='A in force calculation', default="-0.152")

AddSetting(name="GravitationY", comment='Gravitation in the direction of y')
AddSetting(name="GravitationX", comment='Gravitation in the direction of x')
AddSetting(name="GravitationZ", comment='Gravitation in the direction of x')

AddSetting(name="MovingWallVelocity", comment='Velocity of the MovingWall')
AddSetting(name="Density", comment='zonal density', zonal=TRUE)
AddSetting(name="Wetting", comment='wetting factor')

AddGlobal(name="MovingWallForceX", comment='force x')
AddGlobal(name="MovingWallForceY", comment='force y')
AddGlobal(name="MovingWallForceZ", comment='force Z')
AddGlobal(name="Pressure1",        comment='pressure at Obj1')
AddGlobal(name="Pressure2",        comment='pressure at Obj2')
AddGlobal(name="Pressure3",        comment='pressure at Obj3')
AddGlobal(name="Density1",         comment='density at Obj1')
AddGlobal(name="Density2",         comment='density at Obj2')
AddGlobal(name="Density3",         comment='density at Obj3')
