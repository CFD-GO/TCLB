# Setting permissive access policy.
#  * This skips checks of fields being overwritten or read prematurely.
#  * Otherwise the model compilation was failing.
#  * This should be removed if the issue is fixed
SetOptions(permissive.access=TRUE)  ### WARNING



AddDensity( name="f[0]", dx= 0, dy= 0, dz= 0, group="f")
AddDensity( name="f[1]", dx= 1, dy= 0, dz= 0, group="f")
AddDensity( name="f[2]", dx=-1, dy= 0, dz= 0, group="f")
AddDensity( name="f[3]", dx= 0, dy= 1, dz= 0, group="f")
AddDensity( name="f[4]", dx= 0, dy=-1, dz= 0, group="f")
AddDensity( name="f[5]", dx= 0, dy= 0, dz= 1, group="f")
AddDensity( name="f[6]", dx= 0, dy= 0, dz=-1, group="f")
AddDensity( name="f[7]", dx= 1, dy= 1, dz= 0, group="f")
AddDensity( name="f[8]", dx=-1, dy= 1, dz= 0, group="f")
AddDensity( name="f[9]", dx= 1, dy=-1, dz= 0, group="f")
AddDensity( name="f[10]",dx=-1, dy=-1, dz= 0, group="f")
AddDensity( name="f[11]",dx= 1, dy= 0, dz= 1, group="f")
AddDensity( name="f[12]",dx=-1, dy= 0, dz= 1, group="f")
AddDensity( name="f[13]",dx= 1, dy= 0, dz=-1, group="f")
AddDensity( name="f[14]",dx=-1, dy= 0, dz=-1, group="f")
AddDensity( name="f[15]",dx= 0, dy= 1, dz= 1, group="f")
AddDensity( name="f[16]",dx= 0, dy=-1, dz= 1, group="f")
AddDensity( name="f[17]",dx= 0, dy= 1, dz=-1, group="f")
AddDensity( name="f[18]",dx= 0, dy=-1, dz=-1, group="f")
AddDensity( name="f[19]",dx= 1, dy= 1, dz= 1, group="f")
AddDensity( name="f[20]",dx=-1, dy= 1, dz= 1, group="f")
AddDensity( name="f[21]",dx= 1, dy=-1, dz= 1, group="f")
AddDensity( name="f[22]",dx=-1, dy=-1, dz= 1, group="f")
AddDensity( name="f[23]",dx= 1, dy= 1, dz=-1, group="f")
AddDensity( name="f[24]",dx=-1, dy= 1, dz=-1, group="f")
AddDensity( name="f[25]",dx= 1, dy=-1, dz=-1, group="f")
AddDensity( name="f[26]",dx=-1, dy=-1, dz=-1, group="f")

if (!(Options$Isothermal)){
	AddDensity( name="h[0]", dx= 0, dy= 0, dz= 0, group="h")
	AddDensity( name="h[1]", dx= 1, dy= 0, dz= 0, group="h")
	AddDensity( name="h[2]", dx=-1, dy= 0, dz= 0, group="h")
	AddDensity( name="h[3]", dx= 0, dy= 1, dz= 0, group="h")
	AddDensity( name="h[4]", dx= 0, dy=-1, dz= 0, group="h")
	AddDensity( name="h[5]", dx= 0, dy= 0, dz= 1, group="h")
	AddDensity( name="h[6]", dx= 0, dy= 0, dz=-1, group="h")
	AddDensity( name="h[7]", dx= 1, dy= 1, dz= 0, group="h")
	AddDensity( name="h[8]", dx=-1, dy= 1, dz= 0, group="h")
	AddDensity( name="h[9]", dx= 1, dy=-1, dz= 0, group="h")
	AddDensity( name="h[10]",dx=-1, dy=-1, dz= 0, group="h")
	AddDensity( name="h[11]",dx= 1, dy= 0, dz= 1, group="h")
	AddDensity( name="h[12]",dx=-1, dy= 0, dz= 1, group="h")
	AddDensity( name="h[13]",dx= 1, dy= 0, dz=-1, group="h")
	AddDensity( name="h[14]",dx=-1, dy= 0, dz=-1, group="h")
	AddDensity( name="h[15]",dx= 0, dy= 1, dz= 1, group="h")
	AddDensity( name="h[16]",dx= 0, dy=-1, dz= 1, group="h")
	AddDensity( name="h[17]",dx= 0, dy= 1, dz=-1, group="h")
	AddDensity( name="h[18]",dx= 0, dy=-1, dz=-1, group="h")
	AddDensity( name="h[19]",dx= 1, dy= 1, dz= 1, group="h")
	AddDensity( name="h[20]",dx=-1, dy= 1, dz= 1, group="h")
	AddDensity( name="h[21]",dx= 1, dy=-1, dz= 1, group="h")
	AddDensity( name="h[22]",dx=-1, dy=-1, dz= 1, group="h")
	AddDensity( name="h[23]",dx= 1, dy= 1, dz=-1, group="h")
	AddDensity( name="h[24]",dx=-1, dy= 1, dz=-1, group="h")
	AddDensity( name="h[25]",dx= 1, dy=-1, dz=-1, group="h")
	AddDensity( name="h[26]",dx=-1, dy=-1, dz=-1, group="h")
}

#Accessing adjacent nodes
for (d in rows(DensityAll)){
    AddField( name=d$name,  dx=c(1,-1), dy=c(1,-1), dz=c(1,-1) ) }

AddDensity( name="sol", group="Force",parameter=TRUE)
AddDensity( name="uPx", group="Force",parameter=TRUE)
AddDensity( name="uPy", group="Force",parameter=TRUE)
AddDensity( name="uPz", group="Force",parameter=TRUE)

AddQuantity(name="Solid",unit="1")
AddQuantity(name="U",unit="m/s",vector=T)
AddQuantity(name="Rho",unit="kg/m3")

AddSetting(name="omegaF", comment='one over F relaxation time')

AddSetting(name="omegaFplus", comment='one over F+ relaxation time for TRT')
AddSetting(name="omegaFminus", comment='one over F- relaxation time for TRT')

AddDensity( name="localOmegaF", group="l",parameter=TRUE)

AddSetting(name="WallVelocityX", default="0.0", zonal=TRUE, comment='WallVelocity x-direction')
AddSetting(name="WallVelocityY", default="0.0", zonal=TRUE, comment='WallVelocity y-direction')
AddSetting(name="WallVelocityZ", default="0.0", zonal=TRUE, comment='WallVelocity z-direction')

AddSetting(name="InitVelocityX", default="0.0", comment="Initialisation x-velocity")
AddSetting(name="InitVelocityY", default="0.0", comment="Initialisation y-velocity")
AddSetting(name="InitVelocityZ", default="0.0", comment="Initialisation z-velocity")

AddSetting(name="InletPressure", InletDensity='1.0+InletPressure/3', default="0Pa", comment='inlet pressure')
AddSetting(name="InletDensity", default=1, comment='inlet density')
AddSetting(name="Pressure", default="0Pa", comment='Inlet pressure', zonal=TRUE)

AddSetting(name="GravitationX", default=0.0, comment='applied (rho)*GravitationX')
AddSetting(name="GravitationY", default=0.0, comment='applied (rho)*GravitationY')
AddSetting(name="GravitationZ", default=0.0, comment='applied (rho)*GravitationZ')

AddSetting(name="AccelX", default=0.0, comment='body acceleration X')
AddSetting(name="AccelY", default=0.0, comment='body acceleration Y')
AddSetting(name="AccelZ", default=0.0, comment='body acceleration Z')

AddSetting(name="DNx", default = 0, comment='Total nodes in X direction')
AddSetting(name="DNy", default = 0, comment='Total nodes in Y direction')
AddSetting(name="DNz", default = 0, comment='Total nodes in Z direction')
AddGlobal(name="TotalSVF", comment='Total of solids throughout domain')

AddNodeType(name="NWall", group="BOUNDARY")
AddNodeType(name="EWall", group="BOUNDARY")
AddNodeType(name="WWall", group="BOUNDARY")
AddNodeType(name="SWall", group="BOUNDARY")
AddNodeType(name="FWall", group="BOUNDARY")
AddNodeType(name="BWall", group="BOUNDARY")

AddNodeType(name="NPressure", group="BOUNDARY")
AddNodeType(name="EPressure", group="BOUNDARY")
AddNodeType(name="WPressure", group="BOUNDARY")
AddNodeType(name="SPressure", group="BOUNDARY")
AddNodeType(name="FPressure", group="BOUNDARY")
AddNodeType(name="BPressure", group="BOUNDARY")

if (!(Options$Isothermal)){
	AddField('TotEnergy', stencil3d=1, group="h")
	AddField('Temperature', stencil3d=1, group="h")

	AddField('rhoCp',stencil3d=1, group="h")
	AddField('Conductivity', stencil3d=1, group="h")
	AddField('mediaNum', stencil3=1, group="h")


	AddDensity( name="localCv", group="l",parameter=TRUE)
	AddDensity( name="localConductivity", group="l",parameter=TRUE)
	AddDensity( name="localRho", group="l",parameter=TRUE)

	AddDensity( name="localOmegaH", group="l",parameter=TRUE)


	AddQuantity(name="T",unit="K")
	AddQuantity(name="T2",unit="K")
	AddQuantity(name="TotEnergy")
	AddQuantity(name="TE2")
	AddQuantity(name="Cv")

	AddSetting(name="omegaH", comment='one over H relaxation time')
	AddSetting(name="alpha", default=0.16666666, comment='Thermal Diffusivity')


	AddSetting(name="omegaHplus", comment='one over H+ relaxation time for TRT')
	AddSetting(name="omegaHminus", comment='one over H- relaxation time for TRT')

	AddSetting(name="ViscCoeff", default=0.0, comment='Thermoviscous coefficient')
	AddSetting(name="BoussinesqCoeff", default=0.0, comment='Boussinesq force coefficient')

	AddSetting(name='FluidCv', default = 1, comment='Thermal Cv')
	AddSetting(name='FluidRho', default = 1, comment='Material desnsity')
	AddSetting(name="FluidConductivity", default=1, comment='Thermal Conductivity')

	AddSetting(name='SolidCv', default = 1, comment='Thermal Cv')
	AddSetting(name='SolidRho', default = 1, comment='Material density - solid')
	AddSetting(name="SolidConductivity", default=1, comment='Thermal Conductivity')

	AddSetting(name="InitTemperature", default=1, zonal = TRUE, comment='initial temperature')
	AddSetting(name="WallTemperatureGradient", default=0, zonal = TRUE, comment='Gradient of temperature along wall')

	AddSetting(name="MediaNumber", default=1, zonal=TRUE, comment='Media Number')

	AddNodeType(name="Interface", group="ADDITIONALS")
	AddNodeType(name="Med2", group="ADDITIONALS")

}
if (Options$Isothermal) {

AddStage("BaseInit", "Init", save=Fields$group %in% c("f","Force"), load = DensityAll$group %in% c("f","Force"))
AddStage("BaseIteration", "Run", save=Fields$group %in% c("f","Force"), load = DensityAll$group %in% c("f","Force"))
AddStage("CalcF", save=Fields$group == "Force", load = DensityAll$group %in% c("f","Force"), particle=TRUE)

} else {

AddStage("BaseInit", "Init", save=Fields$group %in% c("f","h","Force","l"), load = DensityAll$group %in% c("f","h","Force","l"))
AddStage("BaseIteration", "Run", save=Fields$group %in% c("f","h","Force","l"), load = DensityAll$group %in% c("f","h","Force","l"))
AddStage("CalcF", save=Fields$group == "Force", load = DensityAll$group %in% c("f","Force","l"), particle=TRUE)

}
AddAction("Iteration", c("BaseIteration", "CalcF"))
AddAction("Init", c("BaseInit", "CalcF"))

AddNodeType(name="Solid", group="BOUNDARY")
AddNodeType(name="Wall", group="BOUNDARY")
AddNodeType(name="BGK", group="COLLISION")
