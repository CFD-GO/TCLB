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

# exported to VTK
AddQuantity( name="P",unit="Pa")
AddQuantity( name="U",unit="m/s",vector=T)
AddQuantity( name="nu_app",unit="m2/s")
AddQuantity( name="yield_stat")

AddSetting(name="nu", default=0.16666666, comment='Viscosity')
AddSetting(name="Velocity", default="0m/s", comment='Inlet velocity', zonal=TRUE)
AddSetting(name="Pressure", default="0Pa", comment='Inlet pressure', zonal=TRUE)
	
AddDensity(name="nu_app", dx=0, dy=0, dz=0)
AddDensity(name="yield_stat", dx=0, dy=0, dz=0)

AddSetting(name="GalileanCorrection",default=1.,comment='Galilean correction term')
AddSetting(name="ForceX",default=0, comment='Force force X')
AddSetting(name="ForceY",default=0, comment='Force force Y')
AddSetting(name="ForceZ",default=0, comment='Force force Z')
AddSetting(name="SigmaY",default=0, comment='Yield stress')

AddGlobal(name="Flux", comment='Volume flux', unit="m3/s")

AddNodeType("SymmetryY", "BOUNDARY")
AddNodeType("SymmetryZ", "BOUNDARY")
AddNodeType("TopSymmetry","BOUNDARY")
AddNodeType("BottomSymmetry","BOUNDARY")
AddNodeType("NVelocity", "BOUNDARY")
AddNodeType("SVelocity", "BOUNDARY")
AddNodeType("NPressure", "BOUNDARY")
AddNodeType("SPressure", "BOUNDARY")

# new boundary conditions
AddNodeType("CoutVelocity","BOUNDARY")
AddNodeType("PCoutVelocity","BOUNDARY")
AddNodeType("WVelocityEq","BOUNDARY")
AddNodeType("WVelocityBB","BOUNDARY")

#reporting fluxes
AddNodeType("XYslice1",group="ADDITIONALS")
AddNodeType("XZslice1",group="ADDITIONALS")
AddNodeType("YZslice1",group="ADDITIONALS")
AddNodeType("XYslice2",group="ADDITIONALS")
AddNodeType("XZslice2",group="ADDITIONALS")
AddNodeType("YZslice2",group="ADDITIONALS")

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
        
#Adding terms for supporting time-correlation for synthetic turbulence




