Ave = FALSE
x = c(0,1,-1);
P = expand.grid(x=0:2,y=0:2,z=0:2)
U = expand.grid(x,x,x)

AddDensity(
	name = paste("f",P$x,P$y,P$z,sep=""),
	dx   = U[,1],
	dy   = U[,2],
	dz   = U[,3],
	comment=paste("density F",1:27-1),
	group="f"
)

AddQuantity( name="P",unit="Pa")
AddQuantity( name="U",unit="m/s",vector=T)

AddSetting(name="nu", default=0.16666666, comment='Viscosity')
AddSetting(name="Velocity", default="0m/s", comment='Inlet velocity', zonal=TRUE)
AddSetting(name="Pressure", default="0Pa", comment='Inlet pressure', zonal=TRUE)

AddSetting(name="GalileanCorrection",default=0.,comment='Galilean correction term')
AddSetting(name="ForceX",default=0, comment='Force X')
AddSetting(name="ForceY",default=0,comment='Force Y')
AddSetting(name="ForceZ",default=0,comment='Force Z')

AddGlobal(name="Flux", comment='Volume flux', unit="m3/s")
AddGlobal(name="TotalRho", comment='Total mass', unit="kg")

AddNodeType("SymmetryY", "BOUNDARY")
AddNodeType("SymmetryZ", "BOUNDARY")
AddNodeType("TopSymmetry","BOUNDARY")
AddNodeType("BottomSymmetry","BOUNDARY")
AddNodeType("NVelocity", "BOUNDARY")
AddNodeType("SVelocity", "BOUNDARY")
AddNodeType("NPressure", "BOUNDARY")
AddNodeType("SPressure", "BOUNDARY")