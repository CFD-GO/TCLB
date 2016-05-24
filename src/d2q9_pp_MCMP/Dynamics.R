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

AddDensity( name="g[0]", dx= 0, dy= 0, group="g")
AddDensity( name="g[1]", dx= 1, dy= 0, group="g")
AddDensity( name="g[2]", dx= 0, dy= 1, group="g")
AddDensity( name="g[3]", dx=-1, dy= 0, group="g")
AddDensity( name="g[4]", dx= 0, dy=-1, group="g")
AddDensity( name="g[5]", dx= 1, dy= 1, group="g")
AddDensity( name="g[6]", dx=-1, dy= 1, group="g")
AddDensity( name="g[7]", dx=-1, dy=-1, group="g")
AddDensity( name="g[8]", dx= 1, dy=-1, group="g")

AddField(name="psi_g",  stencil2d=1);
AddField(name="psi_f",  stencil2d=1);

AddStage("BaseIteration", "Run",  save=Fields$group=="f" | Fields$group=="g", load=DensityAll$group=="f" | DensityAll$group =="g")
AddStage("CalcPsi_f"    , save="psi_f", load=DensityAll$group == "f" )
AddStage("CalcPsi_g"    , save="psi_g", load=DensityAll$group == "g" )
AddStage("BaseInit"     , "Init", save=Fields$group=="f" | Fields$group=="g", load=DensityAll$group=="f" | DensityAll$group =="g")

AddAction("Iteration", c("BaseIteration","CalcPsi_f","CalcPsi_g"))
AddAction("Init"     , c("BaseInit",     "CalcPsi_f","CalcPsi_g"))

# Quantities - table of fields that can be exported from the LB lattice (like density, velocity etc)
#  name - name of the field
#  type - C type of the field, "real_t" - for single/double float, and "vector_t" for 3D vector single/double float
# Every field must correspond to a function in "Dynamics.c".
# If one have filed [something] with type [type], one have to define a function: 
# [type] get[something]() { return ...; }

AddQuantity(name="Rho", unit="kg/m3")
AddQuantity(name="Rhof",unit="kg/m3")
AddQuantity(name="Rhog",unit="kg/m3")
AddQuantity(name="P",   unit="Pa")
AddQuantity(name="U",   unit="m/s",vector=T)
AddQuantity(name="A",   unit="1",vector=T)
AddQuantity(name="Ff",  unit="N",vector=T)
AddQuantity(name="Fg",  unit="N",vector=T)

# Settings - table of settings (constants) that are taken from a .xml file
#  name - name of the constant variable
#  comment - additional comment
# You can state that another setting is 'derived' from this one stating for example: omega='1.0/(3*nu + 0.5)'

AddSetting(name="omega_f", comment='one over relaxation time-wet')
AddSetting(name="omega_g", comment='one over relaxation time-dry')
AddSetting(name="nu_f", omega_f='1.0/(3*nu_f + 0.5)', default=1.6666666, comment='viscosity-wet')
AddSetting(name="nu_g", omega_g='1.0/(3*nu_g + 0.5)', default=1.6666666, comment='viscosity-dry')
AddSetting(name="Velocity", default=0, comment='inlet/outlet/init velocity', zonal=T)
AddSetting(name="Pressure", default=0, comment='inlet/outlet/init density', zonal=T)
AddSetting(name="Smag", comment='Smagorinsky constant')

AddSetting(name="SL_U", comment='Shear Layer velocity')
AddSetting(name="SL_lambda", comment='Shear Layer lambda')
AddSetting(name="SL_delta", comment='Shear Layer disturbance')
AddSetting(name="SL_L", comment='Shear Layer length scale')

AddSetting(name="rho_wet", comment='higher density fluid', zonal=T)
AddSetting(name="rho_dry", comment='lower density fluid' , zonal=T)
AddSetting(name="G11", comment='fluid1-fluid1 interaction')
AddSetting(name="G22", comment='fluid2-fluid2 interaction')
AddSetting(name="G12", comment='fluid1-fluid2 interaction')
AddSetting(name="G21", comment='fluid2-fluid1 interaction')
AddSetting(name="Gc", comment='fluid-fluid interation')
AddSetting(name="Gad1", comment='fluid1-wall interation')
AddSetting(name="Gad2", comment='fluid2-wall interation')

# Globals - table of global integrals that can be monitored and optimized
AddGlobal(name="TotalDensity1", comment='quantity of fluid-1', unit='kg/m3')
AddGlobal(name="TotalDensity2", comment='quantity of fluid-2', unit='kg/m3')

AddGlobal(name="PressureLoss", comment='pressure loss', unit="1mPa")
AddGlobal(name="OutletFlux", comment='pressure loss', unit="1m2/s")
AddGlobal(name="InletFlux", comment='pressure loss', unit="1m2/s")

AddNodeType("Smagorinsky", "LES")
AddNodeType("Stab", "ENTROPIC")
