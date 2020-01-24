
ifeq '$(strip $(WITH_CATALYST))' '1'
 PV_LIBS += vtkPVPythonCatalyst
 PV_LIBS += vtkPVCatalyst
 PV_LIBS += vtkCommonDataModel
 PV_LIBS += vtkCommonCore
 PV_INCLUDES += VTK/Common/DataModel
 PV_INCLUDES += VTK/Common/Core
 PV_INCLUDES += CoProcessing/Catalyst
 PV_INCLUDES += ParaViewCore/ServerManager/SMApplication
 PV_INCLUDES += ParaViewCore/ServerManager/Core
 PV_INCLUDES += ParaViewCore/ServerImplementation/Core
 PV_INCLUDES += ParaViewCore/ClientServerCore/Core
 PV_INCLUDES += VTK/Filters/Extraction
 PV_INCLUDES += VTK/Filters/Core
 PV_INCLUDES += VTK/Filters/General
 PV_INCLUDES += VTK/Filters/Statistics
 PV_INCLUDES += VTK/Filters/Parallel
 PV_INCLUDES += VTK/Filters/Geometry
 PV_INCLUDES += VTK/Filters/Modeling
 PV_INCLUDES += VTK/Filters/Sources
 PV_INCLUDES += VTK/Rendering/Core
 PV_INCLUDES += VTK/Utilities/KWIML
 PV_INCLUDES += ParaViewCore/VTKExtensions/Core
 PV_INCLUDES += CoProcessing/PythonCatalyst
 PV_CPPFLAGS += $(addprefix -I $(PV_BUILD_INC)/, $(PV_INCLUDES))
 PV_CPPFLAGS += $(addprefix -I $(PV_SOURCE)/, $(PV_INCLUDES)) -I $(PV_SOURCE)
 PV_CPPFLAGS += -D WITH_CATALYST
 PV_LDFLAGS  += $(addprefix -l, $(addsuffix -$(PV_VERSION),$(PV_LIBS)))
 PV_LDFLAGS  += -Wl,-rpath,$(PV_BUILD)
 PV_LDFLAGS  += -L$(PV_BUILD)

 OPT += $(PV_CPPFLAGS)
 LD_OPT += $(PV_LDFLAGS)
 SOURCE_CU += Catalyst.cpp
 OBJ += Catalyst.o
 HEADERS += Catalyst.h
endif
