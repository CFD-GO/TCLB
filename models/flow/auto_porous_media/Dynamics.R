


IncludedADREModel = TRUE

Qname = 'LinearReaction'
DREs <- ('PHI')
NumberOfODEs = 0
Params  <- c("LinearReactionRate")
D3 = TRUE
D3Q19 = TRUE
QIntegrator = 'Heun'
source("../models/reaction/d2q9_reaction_diffusion_system/Dynamics.R")




x = c(0,1,-1);
P = expand.grid(x=0:2,y=0:2,z=0:2)
U = expand.grid(x,x,x)

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

AddDensity( name="fx",  group="Force", parameter=FALSE)
AddDensity( name="fy",  group="Force", parameter=FALSE)
AddDensity( name="fz",  group="Force", parameter=FALSE)


AddDensity( name="Porosity",  group="Brinkman", parameter=TRUE)
AddDensity( name="Permability",  group="Brinkman", parameter=TRUE)



AddQuantity(name="P",unit="Pa")
AddQuantity(name="U",unit="m/s",vector=T)

AddQuantity(name="Porosity",unit="1")
AddQuantity(name="Permability",unit="1/m2")
AddQuantity(name="BrinkmanForce",unit="N/m3",vector=TRUE)



AddSetting(name="Viscosity", default=0.16666666, comment='Viscosity')
AddSetting(name="Magic", default=3/16, comment='Magic parameter')
AddSetting(name="Velocity", default="0m/s", comment='Inlet velocity', zonal=TRUE)
AddSetting(name="Pressure", default="0Pa", comment='Inlet pressure', zonal=TRUE)

AddSetting(name="GalileanCorrection",default=1.,comment='Galilean correction term')

AddSetting(name="ForceX", default=0, comment='Force force X')
AddSetting(name="ForceY", default=0, comment='Force force Y')
AddSetting(name="ForceZ", default=0, comment='Force force Z')

AddGlobal(name="Flux", comment='Volume flux', unit="m3/s")
AddGlobal(name="Drag", comment='Force exerted on body in X-direction', unit="N")
AddGlobal(name="Lift", comment='Force exerted on body in Z-direction', unit="N")
AddGlobal(name="Lateral", comment='Force exerted on body in Y-direction', unit="N")

AddNodeType(name="NVelocity", group="BOUNDARY")
AddNodeType(name="SVelocity", group="BOUNDARY")
AddNodeType(name="NPressure", group="BOUNDARY")
AddNodeType(name="SPressure", group="BOUNDARY")
AddNodeType(name="EPressure", group="BOUNDARY")
AddNodeType(name="EVelocity", group="BOUNDARY")
AddNodeType(name="Wall", group="BOUNDARY")
AddNodeType(name="WPressure", group="BOUNDARY")
AddNodeType(name="WVelocity", group="BOUNDARY")



#for (f in fname) AddField(f,dx=0,dy=0,dz=0) # Make f accessible also in present node (not only streamed)


AddNodeType(name="Collision", group="COLLISION")
