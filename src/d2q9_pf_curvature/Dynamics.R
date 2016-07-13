# Density - table of variables of LB Node to stream
#  name - variable name to stream
#  dx,dy,dz - direction of streaming
#  comment - additional comment

AddDensity( name="f[0]", dx= 0, dy= 0, group="f")
AddDensity( name="f[1]", dx= 1, dy= 0, group="f")
AddDensity( name="f[2]", dx= 0, dy= 1, group="f")
AddDensity( name="f[3]", dx=-1, dy= 0, group="f")
AddDensity( name="f[4]", dx= 0, dy=-1, group="f")
AddDensity( name="f[5]", dx= 1, dy= 1, group="f")
AddDensity( name="f[6]", dx=-1, dy= 1, group="f")
AddDensity( name="f[7]", dx=-1, dy=-1, group="f")
AddDensity( name="f[8]", dx= 1, dy=-1, group="f")


AddDensity( name="h[0]", dx= 0, dy= 0, group="h")
AddDensity( name="h[1]", dx= 1, dy= 0, group="h")
AddDensity( name="h[2]", dx= 0, dy= 1, group="h")
AddDensity( name="h[3]", dx=-1, dy= 0, group="h")
AddDensity( name="h[4]", dx= 0, dy=-1, group="h")
AddDensity( name="h[5]", dx= 1, dy= 1, group="h")
AddDensity( name="h[6]", dx=-1, dy= 1, group="h")
AddDensity( name="h[7]", dx=-1, dy=-1, group="h")
AddDensity( name="h[8]", dx= 1, dy=-1, group="h")

#AddDensity( name="d[0]", dx= 0, dy= 0, group="d")
#AddDensity( name="d[1]", dx= 1, dy= 0, group="d")
#AddDensity( name="d[2]", dx= 0, dy= 1, group="d")
#AddDensity( name="d[3]", dx=-1, dy= 0, group="d")
#AddDensity( name="d[4]", dx= 0, dy=-1, group="d")
#AddDensity( name="d[5]", dx= 1, dy= 1, group="d")
#AddDensity( name="d[6]", dx=-1, dy= 1, group="d")
#AddDensity( name="d[7]", dx=-1, dy=-1, group="d")
#AddDensity( name="d[8]", dx= 1, dy=-1, group="d")





AddField("phi"       ,stencil2d=1 );

AddStage("BaseIteration", "Run", 
         load=DensityAll$group == "f" | DensityAll$group == "h",# | DensityAll$group == "d",  
         save=Fields$group=="f" | Fields$group=="h",#  | Fields$group=="d"
         ) 
AddStage("CalcPhi", 
         save=Fields$name=="phi" ,  
         load=DensityAll$group == "h"
         )
AddStage("BaseInit", "Init",  save=Fields$group=="f" | Fields$group=="h",#  | Fields$group=="d"
) 


AddAction("Iteration", c("BaseIteration","CalcPhi"))
AddAction("Init", c("BaseInit","CalcPhi"))



# Quantities - table of fields that can be exported from the LB lattice (like density, velocity etc)
#  name - name of the field
#  type - C type of the field, "real_t" - for single/double float, and "vector_t" for 3D vector single/double float
# Every field must correspond to a function in "Dynamics.c".
# If one have filed [something] with type [type], one have to define a function: 
# [type] get[something]() { return ...; }

AddQuantity(name="Rho",unit="kg/m3")
AddQuantity(name="U",unit="m/s",vector=T)

AddQuantity(name="Normal",unit="1/m",vector=T)

AddQuantity(name="PhaseField",unit="1")
AddQuantity(name="Curvature",unit="1")

AddQuantity(name="InterfaceForce", unit="1", vector=T)
#AddQuantity(name="BoundaryForcing", unit="1", vector=T)
#
# Settings - table of settings (constants) that are taken from a .xml file
#  name - name of the constant variable
#  comment - additional comment
# You can state that another setting is 'derived' from this one stating for example: omega='1.0/(3*nu + 0.5)'


AddSetting(name="omega", comment='one over relaxation time')
AddSetting(name="omega_l", comment='one over relaxation time, light phase')
AddSetting(name="nu", omega='1.0/(3*nu + 0.5)', default=0.16666666, comment='viscosity')
AddSetting(name="Velocity", default=0, comment='inlet/outlet/init velocity', zonal=T)
AddSetting(name="Pressure", default=0, comment='inlet/outlet/init density', zonal=T)
AddSetting(name="W", default=1, comment='Anty-diffusivity coeff')
AddSetting(name="M", default=1, comment='Mobility')
AddSetting(name="PhaseField", default=1, comment='Phase Field marker scalar', zonal=T)
AddSetting(name="GravitationX", default=0)
AddSetting(name="GravitationY", default=0)

AddSetting(name="GravitationX_l", default=0)
AddSetting(name="GravitationY_l", default=0)

AddSetting(name="SurfaceTensionDecay", default=100)
AddSetting(name="SurfaceTensionRate", default=0.1)
AddSetting(name="WettingAngle", default=0, zonal=T)
#AddSetting(name="WallDistanceRatio", default=0.5, zonal=T)
# Globals - table of global integrals that can be monitored and optimized

AddGlobal(name="PressureLoss", comment='pressure loss', unit="1mPa")
AddGlobal(name="OutletFlux", comment='pressure loss', unit="1m2/s")
AddGlobal(name="InletFlux", comment='pressure loss', unit="1m2/s")
AddNodeType(name="NSymmetry",group="BOUNDARY")
AddNodeType(name="SSymmetry",group="BOUNDARY")
