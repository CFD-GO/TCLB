Ave = FALSE
x = c(0,1,-1);
P = expand.grid(x=0:2,y=0:2,z=0:2)
U = expand.grid(x,x,x)

fname = paste("f",P$x,P$y,P$z,sep="")
AddDensity(
	name = fname,
	dx   = U[,1],
	dy   = U[,2],
	dz   = U[,3],
	comment=paste("density F",1:27-1),
	group="f"
)

for (f in fname) AddField(f,dx=0,dy=0,dz=0) # Make f accessible also in present node (not only streamed)

for (f in fname[U[,1]<0]) AddField(f,dx=-1,dy=0,dz=0) # Make f accessible in left neighbour for the wierd outlet condition


AddQuantity( name="P",unit="Pa")
AddQuantity( name="U",unit="m/s",vector=T)
# the last one for debugging purposed - will be removed later on

AddSetting(name="nu", default=0.16666666, comment='Viscosity')
AddSetting(name="nubuffer",default=0.01, comment='Viscosity in the buffer layer')
AddSetting(name="Velocity", default="0m/s", comment='Inlet velocity', zonal=TRUE)
AddSetting(name="Pressure", default="0Pa", comment='Inlet pressure', zonal=TRUE)

AddSetting(name="GalileanCorrection",default=1.,comment='Galilean correction term')
AddSetting(name="ForceX",default=0, comment='Force force X')
AddSetting(name="ForceY",default=0,comment='Force force Y')
AddSetting(name="ForceZ",default=0,comment='Force force Z')

AddGlobal(name="Flux", comment='Volume flux', unit="m3/s")

AddNodeType(name="SymmetryY", group="BOUNDARY")
AddNodeType(name="SymmetryZ", group="BOUNDARY")
AddNodeType(name="TopSymmetry", group="BOUNDARY")
AddNodeType(name="BottomSymmetry", group="BOUNDARY")
AddNodeType(name="NVelocity", group="BOUNDARY")
AddNodeType(name="SVelocity", group="BOUNDARY")
AddNodeType(name="NPressure", group="BOUNDARY")
AddNodeType(name="SPressure", group="BOUNDARY")

# new boundary conditions
AddNodeType(name="EOutlet", group="BOUNDARY")
AddNodeType(name="WVelocityEq", group="BOUNDARY")
AddNodeType(name="WVelocityBB", group="BOUNDARY")

#reporting fluxes
AddNodeType(name="XYslice1", group="ADDITIONALS")
AddNodeType(name="XZslice1", group="ADDITIONALS")
AddNodeType(name="YZslice1", group="ADDITIONALS")
AddNodeType(name="XYslice2", group="ADDITIONALS")
AddNodeType(name="XZslice2", group="ADDITIONALS")
AddNodeType(name="YZslice2", group="ADDITIONALS")

# QIBB flag - put as extra for the quadratically-interpolated bounce-back in group HO_BOUNDARY - higher order boundary
AddNodeType(name="QIBB", group="HO_BOUNDARY")

AddGlobal(name="TotalRho", comment='Total mass', unit="kg")

AddGlobal(name="XYvx", comment='Volume flux', unit="m3/s")
AddGlobal(name="XYvy", comment='Volume flux', unit="m3/s")
AddGlobal(name="XYvz", comment='Volume flux', unit="m3/s")
AddGlobal(name="XYrho1", comment='Volume flux', unit="kg/m")
AddGlobal(name="XYrho2", comment='Volume flux', unit="kg/m")
AddGlobal(name="XYarea", comment='Volume flux', unit="m2")

AddGlobal(name="XZvx", comment='Volume flux', unit="m3/s")
AddGlobal(name="XZvy", comment='Volume flux', unit="m3/s")
AddGlobal(name="XZvz", comment='Volume flux', unit="m3/s")
AddGlobal(name="XZrho1", comment='Volume flux', unit="kg/m")
AddGlobal(name="XZrho2", comment='Volume flux', unit="kg/m")
AddGlobal(name="XZarea", comment='Volume flux', unit="m2")

AddGlobal(name="YZvx", comment='Volume flux', unit="m3/s")
AddGlobal(name="YZvy", comment='Volume flux', unit="m3/s")
AddGlobal(name="YZvz", comment='Volume flux', unit="m3/s")
AddGlobal(name="YZrho1", comment='Volume flux', unit="kg/m")
AddGlobal(name="YZrho2", comment='Volume flux', unit="kg/m")
AddGlobal(name="YZarea", comment='Volume flux', unit="m2")

# Adding directional cross-ratios - neglect for a while


AddNodeType(name="EPressure", group="BOUNDARY")
AddNodeType(name="EVelocity", group="BOUNDARY")
AddNodeType(name="Solid", group="BOUNDARY")
AddNodeType(name="Wall", group="BOUNDARY")
AddNodeType(name="WPressure", group="BOUNDARY")
AddNodeType(name="WVelocity", group="BOUNDARY")
AddNodeType(name="MRT", group="COLLISION")
