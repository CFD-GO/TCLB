
U = expand.grid(-1:1,-1:1,-1:1)


#PhaseField lattice
#d3q27
#U_h = U
#d3q15
U_h = rbind(
     c( 0,0,0), 

     c( 1,0,0),      
     c(-1,0,0),      

     c(0, 1,0),      
     c(0,-1,0),      

     c(0,0, 1),      
     c(0,0,-1)      

#     expand.grid(c(-1,1),c(-1,1),c(-1,1))
)


c_sq_h = 0.25
f_eq_order_h = 1


AddDensity(
	name = paste("f",1:27-1,sep=""),
	dx   = U[,1],
	dy   = U[,2],
	dz   = U[,3],
	comment=paste("density F",1:27-1),
	group="f"
)

lU = dim(U_h)[1] 

AddDensity(
	name = paste("h",1:lU-1,sep=""),
	dx   = U_h[,1],
	dy   = U_h[,2],
	dz   = U_h[,3],
	comment=paste("density H",1:lU-1),
	group="h"
)

AddField( name="nw_x", stencil3d=1, group="nw")
AddField( name="nw_y", stencil3d=1, group="nw")
AddField( name="nw_z", stencil3d=1, group="nw")




AddField("phi"       ,stencil3d=1 );

AddStage("BaseIteration", "Run", 
         load=DensityAll$group == "f" | DensityAll$group == "h",# | DensityAll$group == "d",  
         save=Fields$group=="f" | Fields$group=="h" | Fields$group=="nw"
         ) 
AddStage("CalcPhi", 
         save=Fields$name=="phi" ,  
         load=DensityAll$group == "h"
         )
AddStage("BaseInit", "Init",  save=Fields$group=="f" | Fields$group=="h",#  | Fields$group=="d"
) 
AddStage("CalcWallNormall", "CalcNormal",   
         save=Fields$group=="nw",
         fixedPoint=TRUE
         ) 

AddAction("Iteration", c("BaseIteration","CalcPhi"))
AddAction("Init", c("BaseInit","CalcPhi"))

########################################################################



AddQuantity( name="P",unit="Pa")
AddQuantity( name="U",unit="m/s",vector=T)
AddQuantity(name="PhaseField",unit="1")

AddQuantity(name="Normal",unit="1/m",vector=T)
AddQuantity(name="Curvature",unit="1")
AddQuantity(name="InterfaceForce", unit="1", vector=T)

AddQuantity(name="DEBUG", vector=T)


#########################################################################

AddSetting(name="omega_l", comment='one over relaxation time, light phase')
AddSetting(name="nu_l", omega_l='1.0/(3*nu_l + 0.5)', default=0.16666666, comment='viscosity pf=-0.5')

AddSetting(name="omega_h", comment='one over relaxation time, light phase')
AddSetting(name="nu_h", omega_h='1.0/(3*nu_h + 0.5)', default=0.16666666, comment='viscosity pf=0.5')



AddSetting(name="Velocity", default=0, comment='inlet/outlet/init velocity', zonal=T)
AddSetting(name="Pressure", default=0, comment='inlet/outlet/init density', zonal=T)
AddSetting(name="PhaseField", default=0.5, comment='Phase Field marker scalar', zonal=T)



AddSetting(name="IntWidth", default=0.1, comment='Anty-diffusivity coeff')
AddSetting(name="Mobility", default=0.05, comment='Mobility')

AddSetting(name="GravitationX_h", default=0)
AddSetting(name="GravitationY_h", default=0)
AddSetting(name="GravitationX_l", default=0)
AddSetting(name="GravitationY_l", default=0)


AddSetting(name="SurfaceTensionDecay", default=100)
AddSetting(name="SurfaceTensionRate", default=0.1)
AddSetting(name="WettingAngle", default=0, zonal=T)
AddSetting(name="WallAdhesionDecay", default=0, zonal=T)

#########################################################################

AddGlobal(name="Flux", comment='Volume flux', unit="m3/s")

#########################################################################




AddNodeType(name="EPressure", group="BOUNDARY")
AddNodeType(name="Solid", group="BOUNDARY")
AddNodeType(name="Wall", group="BOUNDARY")
AddNodeType(name="WPressure", group="BOUNDARY")
AddNodeType(name="WVelocity", group="BOUNDARY")
AddNodeType(name="MRT", group="COLLISION")
