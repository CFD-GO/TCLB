
AddDensity( name="f[0]", dx= 0, dy= 0, group="f")
AddDensity( name="f[1]", dx= 1, dy= 0, group="f")
AddDensity( name="f[2]", dx= 0, dy= 1, group="f")
AddDensity( name="f[3]", dx=-1, dy= 0, group="f")
AddDensity( name="f[4]", dx= 0, dy=-1, group="f")
AddDensity( name="f[5]", dx= 1, dy= 1, group="f")
AddDensity( name="f[6]", dx=-1, dy= 1, group="f")
AddDensity( name="f[7]", dx=-1, dy=-1, group="f")
AddDensity( name="f[8]", dx= 1, dy=-1, group="f")

AddField("rho_n",stencil2d=1, group="rho_n");

AddStage("BaseIteration", "Run", save=Fields$group == "f"| Fields$group=="nw" , load=DensityAll$group == "f")
AddStage("CalcRhoSC", save=Fields$group == "rho_n",load=DensityAll$group == "f")
AddStage("BaseInit", "Init", save=Fields$group == "f" | Fields$group == "rho_n" , load=DensityAll$group == "f")


AddAction("Init", c("BaseInit"))
AddAction("Iteration", c("BaseIteration","CalcRhoSC"))




AddQuantity(name="Rho", unit="kg/m3");
AddQuantity(name="U", unit="m/s", vector=T);
AddQuantity(name="P", unit="Pa");
AddQuantity(name="F", unit="N", vector=T);
AddQuantity(name="Finternal", unit="N", vector=T);


AddQuantity(name="DEBUG", vector=T);


if (Options$ViscositySmooth) {
    AddSetting(name="omega_l", nu_l="(1./omega_l - 0.5) / 3." , comment='relaxation factor', default=1)
    AddSetting(name="omega_v", nu_v="(1./omega_v - 0.5) / 3." , comment='relaxation factor', default=1)

    AddSetting(name="nu_l", comment='viscosity', default=1/6)
    AddSetting(name="nu_v", comment='viscosity', default=1/6)
} else {
    AddSetting(name="omega_l", comment='relaxation factor', default=1)
    AddSetting(name="omega_v", comment='relaxation factor', default=1)

    AddSetting(name="nu_l", omega_l='1.0/(3*nu_l + 0.5)', comment='viscosity')
    AddSetting(name="nu_v", omega_v='1.0/(3*nu_v + 0.5)', comment='viscosity')
}


AddSetting(name="Magic", default=3/16, comment='Magic parameter')


AddSetting(name="Velocity", default="0m/s", comment='inlet velocity')
AddSetting(name="Temperature", comment='temperature of the liquid/gas')

AddSetting(name="LVRho_phi_dr", default=1, , zonal=TRUE, comment="(wa < 90 <-> LVRho_phi_dr>=1) |  (wa > 90 <-> LVRho_phi_dr>=0) Wetting toning parameter, see DOI: 10.1103/PhysRevE.100.053313")

AddSetting(name="LVRho_ulimit", default=0.01, , zonal=TRUE, comment="Upper limiting value of rho_w see DOI: 10.1103/PhysRevE.100.053313")
AddSetting(name="LVRho_llimit", default=3.2, , zonal=TRUE,  comment="Lower limiting value of rho_w see DOI: 10.1103/PhysRevE.100.053313")


AddSetting(name="Kupershtokh_K", comment='C-S EOS multiplying param', default="0.01")

if (Options$Kupershtokh) {
    AddSetting(name="Kupershtokh_A", comment='A in force calculation - type of stencil (0 for psi*sum(psi(ei)))', default="-0.152")
} else {
    AddSetting(name="Kupershtokh_A", comment='A in force calculation - type of stencil (0 for psi*sum(psi(ei)))', default="0")
}

AddSetting(name="LBL_kappa", comment='kappa - int width parameter (0 is safe value)', default="0")
AddSetting(name="LBL_epsilon0", comment='epsilon0 param - se original paper (2 is safe value)', default="2")



AddSetting(name="GravitationY", comment='Gravitation in the direction of y')
AddSetting(name="GravitationX", comment='Gravitation in the direction of x')
AddSetting(name="MovingWallVelocity", comment='Velocity of the MovingWall')
AddSetting(name="Density", comment='zonal density', zonal=TRUE)
AddSetting(name="Wetting", comment='wetting factor')

AddSetting(name="density_l", comment='density for omega= omega_l')
AddSetting(name="density_v", comment='density for omega= omega_l')
AddSetting(name="nubuffer", comment='Wall buffer density for cumulant')



#AddGlobal(name="MovingWallForceX", comment='force x')
#AddGlobal(name="MovingWallForceY", comment='force y')


AddGlobal(name="Pressure1",        comment='pressure at Obj1')
AddGlobal(name="Pressure2",        comment='pressure at Obj2')
AddGlobal(name="Pressure3",        comment='pressure at Obj3')
AddGlobal(name="Density1",         comment='density at Obj1')
AddGlobal(name="Density2",         comment='density at Obj2')
AddGlobal(name="Density3",         comment='density at Obj3')

AddGlobal(name="SumUsqr",         comment='Sumo o U**2')

AddGlobal(name="WallForce1X", comment='force x')
AddGlobal(name="WallForce1Y", comment='force y')

AddGlobal(name="WallForce2X", comment='force x')
AddGlobal(name="WallForce2Y", comment='force y')

AddGlobal(name="WallForce3X", comment='force x')
AddGlobal(name="WallForce3Y", comment='force y')



AddNodeType(name="NMovingWall", group="BOUNDARY")
AddNodeType(name="MovingWall", group="BOUNDARY")
AddNodeType(name="ESymmetry",group="BOUNDARY")
AddNodeType(name="NSymmetry",group="BOUNDARY")
AddNodeType(name="SSymmetry",group="BOUNDARY")

AddNodeType("SolidBoundary1","OBJECTIVE")
AddNodeType("SolidBoundary2","OBJECTIVE")
AddNodeType("SolidBoundary3","OBJECTIVE")
