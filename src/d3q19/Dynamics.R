MRTMAT = matrix(c(
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
-30,-11,-11,-11,-11,-11,-11,8,8,8,8,8,8,8,8,8,8,8,8,
12,-4,-4,-4,-4,-4,-4,1,1,1,1,1,1,1,1,1,1,1,1,
0,1,-1,0,0,0,0,1,-1,1,-1,1,-1,1,-1,0,0,0,0,
0,-4,4,0,0,0,0,1,-1,1,-1,1,-1,1,-1,0,0,0,0,
0,0,0,1,-1,0,0,1,1,-1,-1,0,0,0,0,1,-1,1,-1,
0,0,0,-4,4,0,0,1,1,-1,-1,0,0,0,0,1,-1,1,-1,
0,0,0,0,0,1,-1,0,0,0,0,1,1,-1,-1,1,1,-1,-1,
0,0,0,0,0,-4,4,0,0,0,0,1,1,-1,-1,1,1,-1,-1,
 0,2,2,-1,-1,-1,-1,1,1,1,1,1,1,1,1,-2,-2,-2,-2,
0,-4,-4,2,2,2,2,1,1,1,1,1,1,1,1,-2,-2,-2,-2,
 0,0,0,1,1,-1,-1,1,1,1,1,-1,-1,-1,-1,0,0,0,0,
0,0,0,-2,-2,2,2,1,1,1,1,-1,-1,-1,-1,0,0,0,0,
 0,0,0,0,0,0,0,1,-1,-1,1,0,0,0,0,0,0,0,0,
 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,-1,-1,1,
 0,0,0,0,0,0,0,0,0,0,0,1,-1,-1,1,0,0,0,0,
0,0,0,0,0,0,0,1,-1,1,-1,-1,1,-1,1,0,0,0,0,
0,0,0,0,0,0,0,-1,-1,1,1,0,0,0,0,1,-1,1,-1,
0,0,0,0,0,0,0,0,0,0,0,1,1,-1,-1,-1,-1,1,1
),19,19)

v = diag(t(MRTMAT) %*% MRTMAT)
MRTMAT.inv = diag(1/v) %*% t(MRTMAT)

selU = c(4,6,8)

AddDensity(
	name = paste("f",0:18,sep=""),
	dx   = MRTMAT[,selU[1]],
	dy   = MRTMAT[,selU[2]],
	dz   = MRTMAT[,selU[3]],
	comment=paste("density F",0:18),
	group="f"
)

AddQuantity( name="P",unit="Pa")
AddQuantity( name="U",unit="m/s",vector=T)

AddSetting(name="omega", comment='One over relaxation time')
AddSetting(name="nu", omega='1.0/(3*nu + 0.5)', default=0.16666666, comment='Viscosity', unit="1m2/s")
AddSetting(name="InletVelocity", default="0m/s", comment='Inlet velocity', unit="1m/s")
AddSetting(name="InletPressure", InletDensity='1.0+InletPressure*3', default="0Pa", comment='Inlet pressure', unit="1Pa")
AddSetting(name="InletDensity", default=1, comment='Inlet density', unit="1kg/m3")

AddSetting(name="ForceX", comment='Force force X')
AddSetting(name="ForceY", comment='Force force Y')
AddSetting(name="ForceZ", comment='Force force Z')

AddGlobal(name="Flux", comment='Volume flux', unit="m3/s")

AddNodeType("XYslice",group="ADDITIONALS");
AddNodeType("XZslice",group="ADDITIONALS");
AddNodeType("YZslice",group="ADDITIONALS");

#AddGlobal(name="XFlux", comment='Volume flux', unit="m3/s")
#AddGlobal(name="YFlux", comment='Volume flux', unit="m3/s")
#AddGlobal(name="ZFlux", comment='Volume flux', unit="m3/s")

AddGlobal(name="XYvx", comment='Volume flux', unit="m3/s")
AddGlobal(name="XYvy", comment='Volume flux', unit="m3/s")
AddGlobal(name="XYvz", comment='Volume flux', unit="m3/s")
AddGlobal(name="XYrho", comment='Volume flux', unit="kg/m")
AddGlobal(name="XYarea", comment='Volume flux', unit="m2")

AddGlobal(name="XZvx", comment='Volume flux', unit="m3/s")
AddGlobal(name="XZvy", comment='Volume flux', unit="m3/s")
AddGlobal(name="XZvz", comment='Volume flux', unit="m3/s")
AddGlobal(name="XZrho", comment='Volume flux', unit="kg/m")
AddGlobal(name="XZarea", comment='Volume flux', unit="m2")

AddGlobal(name="YZvx", comment='Volume flux', unit="m3/s")
AddGlobal(name="YZvy", comment='Volume flux', unit="m3/s")
AddGlobal(name="YZvz", comment='Volume flux', unit="m3/s")
AddGlobal(name="YZrho", comment='Volume flux', unit="kg/m")
AddGlobal(name="YZarea", comment='Volume flux', unit="m2")

AddGlobal(name="VOLvx", comment='Volume flux', unit="m4/s")
AddGlobal(name="VOLvy", comment='Volume flux', unit="m4/s")
AddGlobal(name="VOLvz", comment='Volume flux', unit="m4/s")
AddGlobal(name="VOLpx", comment='Volume flux', unit="mkg/s")
AddGlobal(name="VOLpy", comment='Volume flux', unit="mkg/s")
AddGlobal(name="VOLpz", comment='Volume flux', unit="mkg/s")
AddGlobal(name="VOLrho", comment='Volume flux', unit="kg")
AddGlobal(name="VOLvolume", comment='Volume flux', unit="m3")

AddGlobal(name="MaxV", comment='Max velocity', unit="m3", op="MAX")

