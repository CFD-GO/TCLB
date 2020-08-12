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

if (Options$OutFlow){
    for (d in rows(DensityAll)) {
	AddField( name=d$name, dx=-d$dx-1, dy=-d$dy, dz=d$dz )
    }
    for (d in rows(DensityAll)) {
	AddField( name=d$name, dx=-d$dx, dy=-d$dy-1, dz=-d$dz )
    }
}

# exported to VTK
AddQuantity( name="P",unit="Pa")
AddQuantity( name="U",unit="m/s",vector=T)
AddQuantity( name="nu_app",unit="m2/s")
AddQuantity( name="yield_stat")

AddSetting(name="nu", default=0.16666666, comment='Viscosity')
AddSetting(name="Velocity", default="0m/s", comment='Inlet velocity', zonal=TRUE)
AddSetting(name="Pressure", default="0Pa", comment='Inlet/Outlet pressure', zonal=TRUE) 
# Zou-He BCs do not distinguish between inlet and outlet
	
#apparent viscosity 
AddDensity(name="nu_app", dx=0, dy=0, dz=0) 
# status of the node - 1 if the region is unfielded, 0 if it is yielded (i.e. behaves like a fluid
AddDensity(name="yield_stat", dx=0, dy=0, dz=0) 

AddSetting(name="ForceX",default="0m/s2", comment='Force force X')
AddSetting(name="ForceY",default="0m/s2", comment='Force force Y')
AddSetting(name="ForceZ",default="0m/s2", comment='Force force Z')
# yield stress of the fluid - below this value of stress the fluid behaves as practically solid, yield stress is usually denoted as tau_y or sigma_y
AddSetting(name="YieldStress",default="0Pa", comment='Yield stress') 

AddGlobal(name="Flux", comment='Volume flux', unit="m3/s")

# naming convention used (hopefully consistent with other pieces of code) X: East - West; Y: North-South; Z: Inlet-Outlet (terminology from computer games)
AddNodeType(name="SymmetryY", group="BOUNDARY")
AddNodeType(name="SymmetryZ", group="BOUNDARY")
AddNodeType(name="NVelocity_ZouHe", group="BOUNDARY")
AddNodeType(name="SVelocity_ZouHe", group="BOUNDARY")
AddNodeType(name="EVelocity_ZouHe", group="BOUNDARY")
AddNodeType(name="WVelocity_ZouHe", group="BOUNDARY")
AddNodeType(name="NPressure_ZouHe", group="BOUNDARY")
AddNodeType(name="SPressure_ZouHe", group="BOUNDARY")
AddNodeType(name="EPressure_ZouHe", group="BOUNDARY")
AddNodeType(name="WPressure_ZouHe", group="BOUNDARY")

#reporting fluxes
AddNodeType(name="XYslice1", group="ADDITIONALS")
AddNodeType(name="XZslice1", group="ADDITIONALS")
AddNodeType(name="YZslice1", group="ADDITIONALS")
AddNodeType(name="XYslice2", group="ADDITIONALS")
AddNodeType(name="XZslice2", group="ADDITIONALS")
AddNodeType(name="YZslice2", group="ADDITIONALS")

if (Options$OutFlow) {
    AddNodeType(name="NeumannXP", group="BOUNDARY")
    AddNodeType(name="NeumannYP", group="BOUNDARY")
}

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
AddNodeType(name="Solid", group="BOUNDARY")
AddNodeType(name="Wall", group="BOUNDARY")
AddNodeType(name="MRT", group="COLLISION")
