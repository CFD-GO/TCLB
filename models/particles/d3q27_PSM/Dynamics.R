
Options$particles = Options$NEBB | Options$SUP

iteration_fields_load = c("f")
iteration_fields_save = c("f")

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

if (Options$KL) {
    AddDensity(name="gamma_dot", group="Viscosity")
    AddDensity(name="nu_app", group="Viscosity")

    AddQuantity( name="Shear")
    AddQuantity( name="Nu_app")
    AddQuantity( name="Stress")
    AddQuantity( name="YieldStatus")

    AddSetting( name="Strain_Dim",default=3, comment='Number of dimensions for strain calculation')
    AddSetting( name="eta1", comment='Plastic viscosity component')
    AddSetting( name="eta2", comment='Shear thinning component')
    AddSetting( name="n", comment='Flow behaviour index')
    AddSetting( name="sigmaY", comment='Yield stress')
    AddSetting( name="m", comment="Regularisation parameter")
    AddSetting( name="MaxIter", default=100)
    AddSetting( name="sLim", default=5e-16)
    
    iteration_fields_load = c(iteration_fields_load,"Viscosity")
    iteration_fields_save = c(iteration_fields_save,"Viscosity")
}

AddQuantity(name="U",unit="m/s",vector=T)
AddQuantity(name="Rho",unit="kg/m3")

AddSetting(name="omegaF", comment='one over F relaxation time and initial relaxation time for kl')
AddSetting(name="nu", omegaF='1.0/(3*nu+0.5)', default=0.1, comment='kinetic viscosity in LBM unit')

if (Options$TRT) {
    AddSetting( name="Lambda", comment="TRT Magic Number")
}

AddSetting(name="VelocityX", default="0.0", zonal=TRUE, comment='wall/inlet/outlet velocity x-direction')
AddSetting(name="VelocityY", default="0.0", zonal=TRUE, comment='wall/inlet/outlet velocity y-direction')
AddSetting(name="VelocityZ", default="0.0", zonal=TRUE, comment='wall/inlet/outlet velocity z-direction')

AddSetting(name="Pressure", default="0Pa", comment='Inlet pressure', zonal=TRUE, unit="1Pa")

AddSetting(name="aX_mean", default=0.0, comment='mean of oscillating acceleration X', zonal=TRUE, unit="m/s2")
AddSetting(name="aX_amp", default=0.0, comment='amplitude of oscillating acceleration X', zonal=TRUE, unit="m/s2")
AddSetting(name="aX_freq", default=0.0, comment='frequency of oscillating acceleration', zonal=TRUE, unit="1/s")
AddSetting(name="AccelY", default=0.0, comment='body acceleration Y', zonal=TRUE, unit="m/s2")
AddSetting(name="AccelZ", default=0.0, comment='body acceleration Z', zonal=TRUE, unit="m/s2")

AddNodeType("RegionMeasureX",group="ADDITIONALS")
AddNodeType("RegionMeasureY",group="ADDITIONALS")
AddNodeType("RegionMeasureZ",group="ADDITIONALS")
AddNodeType("PressureMeasure",group="ADDITIONALS")

AddGlobal(name="TotalFluidMomentumX", unit="kgm/s")
AddGlobal(name="TotalFluidMomentumY", unit="kgm/s")
AddGlobal(name="TotalFluidMomentumZ", unit="kgm/s")
AddGlobal(name="TotalFluidMass", unit="kg")
AddGlobal(name="TotalFluidVolume", unit="m3")
AddGlobal(name="FlowRateX", unit="m/s")
AddGlobal(name="FlowRateY", unit="m/s")
AddGlobal(name="FlowRateZ", unit="m/s")
AddGlobal(name="PressureGauge", unit="Pa")

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
            iteration_fields_save = c(iteration_fields_save,"Force")
            AddStage("BaseInit", "Init", save = Fields$group %in% iteration_fields_save, load = FALSE)
            AddStage("BaseIteration", "Run", save = Fields$group %in% iteration_fields_save, load = DensityAll$group %in% iteration_fields_load)
            AddAction("Iteration", "BaseIteration")
            AddAction("Init", "BaseInit")
	} else {
            AddStage("BaseInit", "Init", save = Fields$group %in% iteration_fields_save, load = FALSE)
            AddStage("BaseIteration", "Run", save = Fields$group %in% iteration_fields_save, load = DensityAll$group %in% c(iteration_fields_load,"Force"))
            AddStage("CalcF", save=Fields$group %in% "Force", load = DensityAll$group %in% iteration_fields_load, particle=TRUE)
            AddAction("Iteration", c("BaseIteration", "CalcF"))
            AddAction("Init", c("BaseInit", "CalcF"))
	}
} else {
        AddStage("BaseInit", "Init", save = Fields$group %in% iteration_fields_save, load = FALSE)
        AddStage("BaseIteration", "Run", save = Fields$group %in% iteration_fields_save, load = DensityAll$group %in% iteration_fields_load)
        AddAction("Iteration", "BaseIteration")
        AddAction("Init", "BaseInit")
}
AddNodeType(name="Solid", group="BOUNDARY")
AddNodeType(name="Wall", group="BOUNDARY")
AddNodeType(name="BGK", group="COLLISION")
