
Options$particles = Options$NEBB | Options$SUP

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

#Accessing adjacent nodes
for (d in rows(DensityAll)){
    AddField( name=d$name,  dx=c(1,-1), dy=c(1,-1), dz=c(1,-1) ) }

if (Options$particles) {
    AddDensity( name="sol", group="Force",parameter=TRUE)
    AddDensity( name="uPx", group="Force",parameter=TRUE)
    AddDensity( name="uPy", group="Force",parameter=TRUE)
    AddDensity( name="uPz", group="Force",parameter=TRUE)
    AddQuantity(name="Solid",unit="1")
    AddGlobal(name="TotalSVF", comment='Total of solids throughout domain')
}

AddQuantity(name="U",unit="m/s",vector=T)
AddQuantity(name="Rho",unit="kg/m3")

AddSetting(name="omegaF", comment='one over F relaxation time')
AddSetting(name="nu", omegaF='1.0/(3*nu+0.5)', default=0.1, comment='kinetic viscosity in LBM unit')

if (Options$TRT) {
	AddSetting(name="omegaP", comment='relaxation parameter for odd components in TRT')
}

AddSetting(name="VelocityX", default="0.0", zonal=TRUE, comment='wall/inlet/outlet velocity x-direction')
AddSetting(name="VelocityY", default="0.0", zonal=TRUE, comment='wall/inlet/outlet velocity y-direction')
AddSetting(name="VelocityZ", default="0.0", zonal=TRUE, comment='wall/inlet/outlet velocity z-direction')

AddSetting(name="Pressure", default="0Pa", comment='Inlet pressure', zonal=TRUE, unit="1Pa")

AddSetting(name="AccelX", default=0.0, comment='body acceleration X', unit="m/s2")
AddSetting(name="AccelY", default=0.0, comment='body acceleration Y', unit="m/s2")
AddSetting(name="AccelZ", default=0.0, comment='body acceleration Z', unit="m/s2")

AddGlobal(name="TotalFluidMomentumX", unit="kgm/s")
AddGlobal(name="TotalFluidMomentumY", unit="kgm/s")
AddGlobal(name="TotalFluidMomentumZ", unit="kgm/s")
AddGlobal(name="TotalFluidMass", unit="kg")
AddGlobal(name="TotalFluidVolume", unit="m3")

AddNodeType(name="NVelocity", group="BOUNDARY")
AddNodeType(name="EVelocity", group="BOUNDARY")
AddNodeType(name="WVelocity", group="BOUNDARY")
AddNodeType(name="SVelocity", group="BOUNDARY")
AddNodeType(name="FVelocity", group="BOUNDARY")
AddNodeType(name="BVelocity", group="BOUNDARY")

AddNodeType(name="NPressure", group="BOUNDARY")
AddNodeType(name="EPressure", group="BOUNDARY")
AddNodeType(name="WPressure", group="BOUNDARY")
AddNodeType(name="SPressure", group="BOUNDARY")
AddNodeType(name="FPressure", group="BOUNDARY")
AddNodeType(name="BPressure", group="BOUNDARY")

AddNodeType(name="MovingWall_N", group="BOUNDARY")
AddNodeType(name="MovingWall_S", group="BOUNDARY")

if (Options$particles) {
	if (Options$singlekernel) {
            AddStage("BaseInit", "Init", save = Fields$group %in% c("f","Force"), load = FALSE)
            AddStage("BaseIteration", "Run", save = Fields$group %in% c("f","Force"), load = DensityAll$group %in% "f")
            AddAction("Iteration", "BaseIteration")
            AddAction("Init", "BaseInit")
	} else {
            AddStage("BaseInit", "Init", save = Fields$group %in% "f", load = FALSE)
            AddStage("BaseIteration", "Run", save = Fields$group %in% "f", load = DensityAll$group %in% c("f","Force"))
            AddStage("CalcF", save=Fields$group %in% "Force", load = DensityAll$group %in% "f", particle=TRUE)
            AddAction("Iteration", c("BaseIteration", "CalcF"))
            AddAction("Init", c("BaseInit", "CalcF"))
	}
} else {
        AddStage("BaseInit", "Init", save = Fields$group %in% "f", load = FALSE)
        AddStage("BaseIteration", "Run", save = Fields$group %in% "f", load = DensityAll$group %in% "f")
        AddAction("Iteration", "BaseIteration")
        AddAction("Init", "BaseInit")
}
