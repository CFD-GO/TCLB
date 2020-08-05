
AddDensity(name="f0",  dx=0 , dy=0 , dz=0  , group="f" )
AddDensity(name="f1",  dx=1 , dy=0 , dz=0  , group="f" )
AddDensity(name="f2",  dx=0 , dy=1 , dz=0  , group="f" )
AddDensity(name="f3",  dx=-1, dy=0 , dz=0  , group="f" )
AddDensity(name="f4",  dx=0 , dy=-1, dz=0  , group="f" )
AddDensity(name="f5",  dx=0 , dy=0 , dz=1  , group="f" )
AddDensity(name="f6",  dx=0 , dy=0 , dz=-1 , group="f" )
AddDensity(name="f7",  dx=1 , dy=1 , dz=0  , group="f" )
AddDensity(name="f8",  dx=-1, dy=1 , dz=0  , group="f" )
AddDensity(name="f9",  dx=-1, dy=-1, dz=0  , group="f" )
AddDensity(name="f10", dx=1 , dy=-1, dz=0  , group="f" )
AddDensity(name="f11", dx=1 , dy=0 , dz=1  , group="f" )
AddDensity(name="f12", dx=0 , dy=1 , dz=1  , group="f" )
AddDensity(name="f13", dx=-1, dy=0 , dz=1  , group="f" )
AddDensity(name="f14", dx=0 , dy=-1, dz=1  , group="f" )
AddDensity(name="f15", dx=1 , dy=0 , dz=-1 , group="f" )
AddDensity(name="f16", dx=0 , dy=1 , dz=-1 , group="f" )
AddDensity(name="f17", dx=-1, dy=0 , dz=-1 , group="f" )
AddDensity(name="f18", dx=0 , dy=-1, dz=-1 , group="f" )
AddDensity(name="f19", dx=1 , dy=1 , dz=1  , group="f" )
AddDensity(name="f20", dx=-1, dy=1 , dz=1  , group="f" )
AddDensity(name="f21", dx=-1, dy=-1, dz=1  , group="f" )
AddDensity(name="f22", dx=1 , dy=-1, dz=1  , group="f" )
AddDensity(name="f23", dx=1 , dy=1 , dz=-1 , group="f" )
AddDensity(name="f24", dx=-1, dy=1 , dz=-1 , group="f" )
AddDensity(name="f25", dx=-1, dy=-1, dz=-1 , group="f" )
AddDensity(name="f26", dx=1 , dy=-1, dz=-1 , group="f" )

if (Options$OutFlow){
    for (d in rows(DensityAll)) {
        AddField( name=d$name, dx=-d$dx+c(-1,1), dy=-d$dy, dz=-d$dz)
        AddField( name=d$name, dx=-d$dx, dy=-d$dy+c(-1,1), dz=-d$dz)
        AddField( name=d$name, dx=-d$dx, dy=-d$dy, dz=-d$dz+c(-1,1))
    }
}

AddDensity(name="gamma_dot")
AddDensity(name="nu_app")
AddDensity(name="Omega")
for (i in c("xx","xy","yz","yy","zx","zz")){
    AddDensity( name=paste("D",i,sep=""))
}
AddDensity(name="Iter")
AddDensity(name="lambda_even")
AddDensity(name="lambda_odd")

AddQuantity( name="U",unit="m/s", vector=TRUE )
AddQuantity( name="Rho",unit="kg/m3")
AddQuantity( name="Shear")#,unit="1/s")
AddQuantity( name="Nu_app")
AddQuantity( name="Stress")
AddQuantity( name="YieldStatus")
for (i in c("xx","xy","yz","yy","zx","zz")){
    AddQuantity( name=paste("D",i,sep=""))
}
AddQuantity( name="Pressure")
AddQuantity( name="Iterations")
AddQuantity( name="Lambda_even")
AddQuantity( name="Lambda_odd")

AddSetting( name="VelocityX",default=0, comment='inlet/outlet/init velocity', zonal=TRUE)
AddSetting( name="VelocityY",default=0, comment='inlet/outlet/init velocity', zonal=TRUE)
AddSetting( name="GravitationX",default=0, comment='body/external acceleration', zonal=TRUE)
AddSetting( name="GravitationY",default=0, comment='body/external acceleration', zonal=TRUE)
AddSetting( name="GravitationZ",default=0, comment='body/external acceleration', zonal=TRUE)
AddSetting( name="Pressure", default=0.3333, comment='Pressure for boundary condition', zonal=TRUE)
AddSetting( name="Density",default=1, comment='Density')

AddSetting( name="Strain_Dim",default=3, comment='Number of dimensions for strain calculation')

#For Varying BC
AddSetting( name="deltaP", comment="half range of pressure fluctuations", zonal="TRUE")
AddSetting( name="Period", comment="Period of pressure fluctuations", zonal="TRUE")
AddSetting( name="Pmax", comment="Heartbeat Pmax", zonal="TRUE")

AddSetting( name="eta1", comment='Plastic viscosity component')
AddSetting( name="eta2", comment='Shear thinning component')
AddSetting( name="sigmaY", comment='Yield stress')

AddSetting( name="m", comment="Regularisation parameter")
AddSetting( name="Lambda", comment="TRT Magic Number")
AddSetting( name="MaxIter", default=100)
AddSetting( name="sLim", default=5e-16)

AddNodeType(name="BGK", group="COLLISION")
AddNodeType(name="TRT", group="COLLISION")
AddNodeType(name="Solid", group="BOUNDARY")
AddNodeType(name="Wall", group="BOUNDARY")
AddNodeType(name="PressureXP", group="BOUNDARY")
AddNodeType(name="PressureXN", group="BOUNDARY")
AddNodeType(name="PressureSinXN", group="BOUNDARY")
AddNodeType(name="PressureCosXN", group="BOUNDARY")
AddNodeType(name="PressureHBXN", group="BOUNDARY")
AddNodeType(name="ExtendedBdy", group="ADDITIONALS")

if (Options$OutFlow){
    AddNodeType(name="NeumannXP", group="BOUNDARY")
    AddNodeType(name="NeumannXN", group="BOUNDARY")
    AddNodeType(name="NeumannYP", group="BOUNDARY")
    AddNodeType(name="NeumannYN", group="BOUNDARY")
    AddNodeType(name="NeumannZP", group="BOUNDARY")
    AddNodeType(name="NeumannZN", group="BOUNDARY")
}

AddGlobal(name="VelocityMax",op="MAX")

AddNodeType(name="LogP", group="ADDITIONALS")
AddGlobal(name="Log_Ux")
AddGlobal(name="Log_Uy")
AddGlobal(name="Log_Uz")
AddGlobal(name="Log_P")
AddGlobal(name="Log_rho")


