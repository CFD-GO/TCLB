


IncludedADREModel = TRUE

Qname = 'DissolutionReaction_ForImplicitSteadyState'
DREs <- ('PHI')
NumberOfODEs = 0
Params  <- c("LinearReactionRate")

if (Options$d2q9) {
    D3 = FALSE
} else {
    D3 = TRUE
    D3Q19 = FALSE #For advection solver
}
QIntegrator = 'Heun'
#QIntegrator = 'Trapezoid'
source("../models/reaction/d2q9_reaction_diffusion_system/Dynamics.R")

if (Options$d2q9) {
	x = c(0,1,-1);
    P = expand.grid(x=0:2,y=0:2,z=0)
    U = expand.grid(x,x,0)
} else {
    x = c(0,1,-1);
    P = expand.grid(x=0:2,y=0:2,z=0:2)
    U = expand.grid(x,x,x)
}



f_sel = rep(TRUE,nrow(U))

if (Options$d3q19) {
	f_sel = rowSums(abs(U)) < 3
}

P=P[f_sel,]
U=U[f_sel,]
fname = paste("f",P$x,P$y,P$z,sep="")

AddDensity(
	name = fname,
	dx   = U[,1],
	dy   = U[,2],
	dz   = U[,3],
	comment=paste("density",fname),
	group="f"
)

#AddDensity( name="fx",  group="Force", parameter=FALSE)
#AddDensity( name="fy",  group="Force", parameter=FALSE)
#AddDensity( name="fz",  group="Force", parameter=FALSE)


AddDensity( name="InitialPorosity",  group="Brinkman", parameter=TRUE)
AddDensity( name="InitialPermability",  group="Brinkman", parameter=TRUE)

AddDensity( name="Porosity",  group="Brinkman", parameter=TRUE)
AddDensity( name="Permability",  group="Brinkman", parameter=TRUE)

AddQuantity(name="P",unit="Pa")
AddQuantity(name="U",unit="m/s",vector=T)

AddQuantity(name="Porosity",unit="1")
AddQuantity(name="Permability",unit="1/m2")
AddQuantity(name="ReactiveFlux",unit="kg/s")
AddQuantity(name="BrinkmanForce",unit="N/m3",vector=TRUE)


AddStage(name="InitFromExternal", load.densities=TRUE, save.fields=TRUE)
AddAction(name="InitFromExternalAction", "InitFromExternal")

AddStage(name="GlobasPorosityDissolutionTimeStep", load.densities=DensityAll$group == "Brinkman", save.fields=Fields$group == "Brinkman")
AddAction(name="GlobasPorosityDissolutionTimeStepAction", "GlobasPorosityDissolutionTimeStep")


AddSetting(name="Viscosity", default=0.16666666, comment='Viscosity')
AddSetting(name="Magic", default=3/16, comment='Magic parameter')
AddSetting(name="Velocity", default="0m/s", comment='Inlet velocity', zonal=TRUE)
AddSetting(name="Pressure", default="0Pa", comment='Inlet pressure', zonal=TRUE)

AddSetting( name="SolidFluidReactionsRate", default="1", comment='Coefficent between fluid and solid mass flux, with dt ratio', zonal=TRUE)
AddSetting( name="ImpliciteReactionIntegration", default="0", comment='Fixed point implicit solution in terms of DPer/Dt.', zonal=FALSE)
AddSetting( name="KarmanKozenyCoefficient", default="1", comment='Permability = KKC * Porosity^3 / ((1-Porosity)^2+1E-8)', zonal=FALSE)

AddSetting(name="GalileanCorrection",default=1.,comment='Galilean correction term')

AddSetting(name="ForceX", default=0, comment='Force force X')
AddSetting(name="ForceY", default=0, comment='Force force Y')
AddSetting(name="ForceZ", default=0, comment='Force force Z')

AddGlobal(name="Flux", comment='Volume flux')
AddGlobal(name="Concentration", comment='Volume flux')

AddGlobal(name="PressureLoss", comment='pressure loss')
AddGlobal(name="OutletFlux", comment='pressure loss')
AddGlobal(name="InletFlux", comment='pressure loss')

# AddNodeType(name="NVelocity", group="BOUNDARY")
# AddNodeType(name="SVelocity", group="BOUNDARY")
# AddNodeType(name="NPressure", group="BOUNDARY")
# AddNodeType(name="SPressure", group="BOUNDARY")
# AddNodeType(name="EPressure", group="BOUNDARY")
# AddNodeType(name="EVelocity", group="BOUNDARY")
AddNodeType(name="Wall", group="BOUNDARY")
if (Options$FlowInX || Options$d2q9) {
    AddNodeType(name="WDirichlet", group="BOUNDARY")
    AddNodeType(name="ENeuman", group="BOUNDARY")
}

if (Options$FlowInZ) { 
    AddNodeType(name="IDirichlet", group="BOUNDARY")
    AddNodeType(name="ONeuman", group="BOUNDARY")
}

for (d in rows(DensityAll)) {
        if (Options$FlowInX || Options$d2q9) { AddField( name=d$name, dx=-d$dx-1, dy=-d$dy, dz=-d$dz ) }
        if (Options$FlowInZ) { AddField( name=d$name, dx=-d$dx, dy=-d$dy, dz=-d$dz-1 ) }
}

# AddNodeType(name="WVelocity", group="BOUNDARY")

# AddNodeType(name="W_PHI_DIRICHLET", group="BOUNDARY")
# AddNodeType(name="E_PHI_DIRICHLET", group="BOUNDARY")

#for (f in fname) AddField(f,dx=0,dy=0,dz=0) # Make f accessible also in present node (not only streamed)


AddNodeType(name="Collision", group="COLLISION")

AddNodeType(name="Inlet", group="OBJECTIVE")
AddNodeType(name="Outlet", group="OBJECTIVE")