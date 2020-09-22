
#--------------------------------------------------------------

# Global timestep output options
timeStepToStartOutputAt=0
forceOutputAtFirstCall=False

# Global screenshot output options
imageFileNamePadding=0
rescale_lookuptable=False

# Whether or not to request specific arrays from the adaptor.
requestSpecificArrays=False

# a root directory under which all Catalyst output goes
rootDirectory=''

# makes a cinema D index table
make_cinema_table=False

#--------------------------------------------------------------
# Code generated from cpstate.py to create the CoProcessor.
# paraview version 5.8.0
#--------------------------------------------------------------

from paraview.simple import *
from paraview import coprocessing

# ----------------------- CoProcessor definition -----------------------

def CreateCoProcessor():
  def _CreatePipeline(coprocessor, datadescription):
    class Pipeline:
      # state file generated using paraview version 5.8.0

      # ----------------------------------------------------------------
      # setup views used in the visualization
      # ----------------------------------------------------------------

      # trace generated using paraview version 5.8.0
      #
      # To ensure correct image size when batch processing, please search 
      # for and uncomment the line `# renderView*.ViewSize = [*,*]`

      #### disable automatic camera reset on 'Show'
      paraview.simple._DisableFirstRenderCameraReset()

      # get the material library
      materialLibrary1 = GetMaterialLibrary()

      # Create a new 'Render View'
      renderView1 = CreateView('RenderView')
      renderView1.ViewSize = [1174, 796]
      renderView1.InteractionMode = '2D'
      renderView1.AxesGrid = 'GridAxes3DActor'
      renderView1.CenterOfRotation = [400.0, 50.0, 100.0]
      renderView1.StereoType = 'Crystal Eyes'
      renderView1.CameraPosition = [396.1243168145181, -1552.1597116387295, 35.42488410229263]
      renderView1.CameraFocalPoint = [396.1243168145181, 50.0, 35.42488410229263]
      renderView1.CameraViewUp = [0.0, 0.0, 1.0]
      renderView1.CameraFocalDisk = 1.0
      renderView1.CameraParallelScale = 283.22481160307757
      renderView1.BackEnd = 'OSPRay raycaster'
      renderView1.OSPRayMaterialLibrary = materialLibrary1

      # register the view with coprocessor
      # and provide it with information such as the filename to use,
      # how frequently to write the images, etc.
      coprocessor.RegisterView(renderView1,
          filename='RenderView1_%t.png', freq=1, fittoscreen=0, magnification=1, width=1174, height=796, cinema={}, compression=5)
      renderView1.ViewTime = datadescription.GetTime()

      SetActiveView(None)

      # ----------------------------------------------------------------
      # setup view layouts
      # ----------------------------------------------------------------

      # create new layout object 'Layout #1'
      layout1 = CreateLayout(name='Layout #1')
      layout1.AssignView(0, renderView1)

      # ----------------------------------------------------------------
      # restore active view
      SetActiveView(renderView1)
      # ----------------------------------------------------------------

      # ----------------------------------------------------------------
      # setup the data processing pipelines
      # ----------------------------------------------------------------

      # create a new 'XDMF Reader'
      # create a producer from a simulation input
      extendedRun_mm_200623_325_HDF5_00 = coprocessor.CreateProducer(datadescription, 'input')

      # create a new 'Contour'
      contour4 = Contour(Input=extendedRun_mm_200623_325_HDF5_00)
      contour4.ContourBy = ['POINTS', 'PhaseField']
      contour4.Isosurfaces = [0.5]
      contour4.PointMergeMethod = 'Uniform Binning'

      # create a new 'Contour'
      contour1 = Contour(Input=extendedRun_mm_200623_325_HDF5_00)
      contour1.ContourBy = ['POINTS', 'PhaseField']
      contour1.Isosurfaces = [0.5]
      contour1.PointMergeMethod = 'Uniform Binning'

      # create a new 'Slice'
      slice1 = Slice(Input=extendedRun_mm_200623_325_HDF5_00)
      slice1.SliceType = 'Plane'
      slice1.HyperTreeGridSlicer = 'Plane'
      slice1.SliceOffsetValues = [0.0]

      # init the 'Plane' selected for 'SliceType'
      slice1.SliceType.Origin = [400.0, 50.0, 100.0]
      slice1.SliceType.Normal = [0.0, 0.0, 1.0]

      # init the 'Plane' selected for 'HyperTreeGridSlicer'
      slice1.HyperTreeGridSlicer.Origin = [400.0, 50.0, 100.0]

      # create a new 'Contour'
      contour3 = Contour(Input=slice1)
      contour3.ContourBy = ['POINTS', 'T']
      contour3.Isosurfaces = [0.000473485, 0.040296065526315795, 0.08011864605263158, 0.11994122657894737, 0.15976380710526317, 0.19958638763157896, 0.23940896815789475, 0.2792315486842105, 0.31905412921052634, 0.35887670973684216, 0.3986992902631579, 0.4385218707894737, 0.4783444513157895, 0.5181670318421053, 0.557989612368421, 0.5978121928947369, 0.6376347734210527, 0.6774573539473685, 0.7172799344736843, 0.757102515]
      contour3.PointMergeMethod = 'Uniform Binning'

      # create a new 'Slice'
      slice2 = Slice(Input=extendedRun_mm_200623_325_HDF5_00)
      slice2.SliceType = 'Plane'
      slice2.HyperTreeGridSlicer = 'Plane'
      slice2.SliceOffsetValues = [0.0]

      # init the 'Plane' selected for 'SliceType'
      slice2.SliceType.Origin = [400.0, 50.0, 100.0]
      slice2.SliceType.Normal = [0.0, 1.0, 0.0]

      # init the 'Plane' selected for 'HyperTreeGridSlicer'
      slice2.HyperTreeGridSlicer.Origin = [400.0, 50.0, 100.0]

      # create a new 'Contour'
      contour2 = Contour(Input=slice2)
      contour2.ContourBy = ['POINTS', 'T']
      contour2.Isosurfaces = [0.000473485, 0.040296065526315795, 0.08011864605263158, 0.11994122657894737, 0.15976380710526317, 0.19958638763157896, 0.23940896815789475, 0.2792315486842105, 0.31905412921052634, 0.35887670973684216, 0.3986992902631579, 0.4385218707894737, 0.4783444513157895, 0.5181670318421053, 0.557989612368421, 0.5978121928947369, 0.6376347734210527, 0.6774573539473685, 0.7172799344736843, 0.757102515]
      contour2.PointMergeMethod = 'Uniform Binning'

      # ----------------------------------------------------------------
      # setup the visualization in view 'renderView1'
      # ----------------------------------------------------------------

      # show data from extendedRun_mm_200623_325_HDF5_00
      extendedRun_mm_200623_325_HDF5_00Display = Show(extendedRun_mm_200623_325_HDF5_00, renderView1, 'UniformGridRepresentation')

      # trace defaults for the display properties.
      extendedRun_mm_200623_325_HDF5_00Display.Representation = 'Outline'
      extendedRun_mm_200623_325_HDF5_00Display.ColorArrayName = ['POINTS', '']
      extendedRun_mm_200623_325_HDF5_00Display.OSPRayScaleArray = 'T'
      extendedRun_mm_200623_325_HDF5_00Display.OSPRayScaleFunction = 'PiecewiseFunction'
      extendedRun_mm_200623_325_HDF5_00Display.SelectOrientationVectors = 'U'
      extendedRun_mm_200623_325_HDF5_00Display.ScaleFactor = 79.9
      extendedRun_mm_200623_325_HDF5_00Display.SelectScaleArray = 'T'
      extendedRun_mm_200623_325_HDF5_00Display.GlyphType = 'Arrow'
      extendedRun_mm_200623_325_HDF5_00Display.GlyphTableIndexArray = 'T'
      extendedRun_mm_200623_325_HDF5_00Display.GaussianRadius = 3.995
      extendedRun_mm_200623_325_HDF5_00Display.SetScaleArray = ['POINTS', 'T']
      extendedRun_mm_200623_325_HDF5_00Display.ScaleTransferFunction = 'PiecewiseFunction'
      extendedRun_mm_200623_325_HDF5_00Display.OpacityArray = ['POINTS', 'T']
      extendedRun_mm_200623_325_HDF5_00Display.OpacityTransferFunction = 'PiecewiseFunction'
      extendedRun_mm_200623_325_HDF5_00Display.DataAxesGrid = 'GridAxesRepresentation'
      extendedRun_mm_200623_325_HDF5_00Display.PolarAxes = 'PolarAxesRepresentation'
      extendedRun_mm_200623_325_HDF5_00Display.ScalarOpacityUnitDistance = 3.309179684253434
      extendedRun_mm_200623_325_HDF5_00Display.IsosurfaceValues = [0.378788]
      extendedRun_mm_200623_325_HDF5_00Display.SliceFunction = 'Plane'
      extendedRun_mm_200623_325_HDF5_00Display.Slice = 99

      # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
      extendedRun_mm_200623_325_HDF5_00Display.ScaleTransferFunction.Points = [0.000473485, 0.0, 0.5, 0.0, 0.757102515, 1.0, 0.5, 0.0]

      # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
      extendedRun_mm_200623_325_HDF5_00Display.OpacityTransferFunction.Points = [0.000473485, 0.0, 0.5, 0.0, 0.757102515, 1.0, 0.5, 0.0]

      # init the 'Plane' selected for 'SliceFunction'
      extendedRun_mm_200623_325_HDF5_00Display.SliceFunction.Origin = [400.0, 50.0, 100.0]

      # show data from contour1
      contour1Display = Show(contour1, renderView1, 'GeometryRepresentation')

      # trace defaults for the display properties.
      contour1Display.Representation = 'Surface'
      contour1Display.ColorArrayName = ['POINTS', '']
      contour1Display.OSPRayScaleArray = 'PhaseField'
      contour1Display.OSPRayScaleFunction = 'PiecewiseFunction'
      contour1Display.SelectOrientationVectors = 'U'
      contour1Display.ScaleFactor = 6.49833984375
      contour1Display.SelectScaleArray = 'PhaseField'
      contour1Display.GlyphType = 'Arrow'
      contour1Display.GlyphTableIndexArray = 'PhaseField'
      contour1Display.GaussianRadius = 0.32491699218750003
      contour1Display.SetScaleArray = ['POINTS', 'PhaseField']
      contour1Display.ScaleTransferFunction = 'PiecewiseFunction'
      contour1Display.OpacityArray = ['POINTS', 'PhaseField']
      contour1Display.OpacityTransferFunction = 'PiecewiseFunction'
      contour1Display.DataAxesGrid = 'GridAxesRepresentation'
      contour1Display.PolarAxes = 'PolarAxesRepresentation'

      # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
      contour1Display.ScaleTransferFunction.Points = [0.5, 0.0, 0.5, 0.0, 0.5001220703125, 1.0, 0.5, 0.0]

      # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
      contour1Display.OpacityTransferFunction.Points = [0.5, 0.0, 0.5, 0.0, 0.5001220703125, 1.0, 0.5, 0.0]

      # show data from contour2
      contour2Display = Show(contour2, renderView1, 'GeometryRepresentation')

      # get color transfer function/color map for 'T'
      tLUT = GetColorTransferFunction('T')
      tLUT.AutomaticRescaleRangeMode = 'Never'
      tLUT.RGBPoints = [0.0, 0.0, 0.0, 0.0, 0.4000000000000001, 0.901960784314, 0.0, 0.0, 0.8000000000000002, 0.901960784314, 0.901960784314, 0.0, 1.0, 1.0, 1.0, 1.0]
      tLUT.ColorSpace = 'RGB'
      tLUT.NanColor = [0.0, 0.498039215686, 1.0]
      tLUT.ScalarRangeInitialized = 1.0

      # trace defaults for the display properties.
      contour2Display.Representation = 'Surface'
      contour2Display.ColorArrayName = ['POINTS', 'T']
      contour2Display.LookupTable = tLUT
      contour2Display.OSPRayScaleArray = 'T'
      contour2Display.OSPRayScaleFunction = 'PiecewiseFunction'
      contour2Display.SelectOrientationVectors = 'U'
      contour2Display.ScaleFactor = 71.02222213745118
      contour2Display.SelectScaleArray = 'T'
      contour2Display.GlyphType = 'Arrow'
      contour2Display.GlyphTableIndexArray = 'T'
      contour2Display.GaussianRadius = 3.5511111068725585
      contour2Display.SetScaleArray = ['POINTS', 'T']
      contour2Display.ScaleTransferFunction = 'PiecewiseFunction'
      contour2Display.OpacityArray = ['POINTS', 'T']
      contour2Display.OpacityTransferFunction = 'PiecewiseFunction'
      contour2Display.DataAxesGrid = 'GridAxesRepresentation'
      contour2Display.PolarAxes = 'PolarAxesRepresentation'

      # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
      contour2Display.ScaleTransferFunction.Points = [0.084543377161026, 0.0, 0.5, 0.0, 0.7571024894714355, 1.0, 0.5, 0.0]

      # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
      contour2Display.OpacityTransferFunction.Points = [0.084543377161026, 0.0, 0.5, 0.0, 0.7571024894714355, 1.0, 0.5, 0.0]

      # show data from slice1
      slice1Display = Show(slice1, renderView1, 'GeometryRepresentation')

      # get color transfer function/color map for 'U'
      uLUT = GetColorTransferFunction('U')
      uLUT.AutomaticRescaleRangeMode = 'Never'
      uLUT.RGBPoints = [0.0, 0.231373, 0.298039, 0.752941, 0.00025, 0.865003, 0.865003, 0.865003, 0.0005, 0.705882, 0.0156863, 0.14902]
      uLUT.ScalarRangeInitialized = 1.0

      # trace defaults for the display properties.
      slice1Display.Representation = 'Surface'
      slice1Display.ColorArrayName = ['POINTS', 'U']
      slice1Display.LookupTable = uLUT
      slice1Display.Position = [0.0, 0.0, -25.0]
      slice1Display.Orientation = [-90.0, 0.0, 0.0]
      slice1Display.OSPRayScaleArray = 'T'
      slice1Display.OSPRayScaleFunction = 'PiecewiseFunction'
      slice1Display.SelectOrientationVectors = 'U'
      slice1Display.ScaleFactor = 79.9
      slice1Display.SelectScaleArray = 'T'
      slice1Display.GlyphType = 'Arrow'
      slice1Display.GlyphTableIndexArray = 'T'
      slice1Display.GaussianRadius = 3.995
      slice1Display.SetScaleArray = ['POINTS', 'T']
      slice1Display.ScaleTransferFunction = 'PiecewiseFunction'
      slice1Display.OpacityArray = ['POINTS', 'T']
      slice1Display.OpacityTransferFunction = 'PiecewiseFunction'
      slice1Display.DataAxesGrid = 'GridAxesRepresentation'
      slice1Display.PolarAxes = 'PolarAxesRepresentation'

      # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
      slice1Display.ScaleTransferFunction.Points = [0.000473485, 0.0, 0.5, 0.0, 0.757102515, 1.0, 0.5, 0.0]

      # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
      slice1Display.OpacityTransferFunction.Points = [0.000473485, 0.0, 0.5, 0.0, 0.757102515, 1.0, 0.5, 0.0]

      # init the 'PolarAxesRepresentation' selected for 'PolarAxes'
      slice1Display.PolarAxes.Translation = [0.0, 0.0, -25.0]
      slice1Display.PolarAxes.Orientation = [-90.0, 0.0, 0.0]

      # show data from contour3
      contour3Display = Show(contour3, renderView1, 'GeometryRepresentation')

      # trace defaults for the display properties.
      contour3Display.Representation = 'Surface'
      contour3Display.ColorArrayName = ['POINTS', 'T']
      contour3Display.LookupTable = tLUT
      contour3Display.Position = [0.0, 0.0, -25.0]
      contour3Display.Orientation = [-90.0, 0.0, 0.0]
      contour3Display.OSPRayScaleArray = 'T'
      contour3Display.OSPRayScaleFunction = 'PiecewiseFunction'
      contour3Display.SelectOrientationVectors = 'U'
      contour3Display.ScaleFactor = 71.02222213745118
      contour3Display.SelectScaleArray = 'T'
      contour3Display.GlyphType = 'Arrow'
      contour3Display.GlyphTableIndexArray = 'T'
      contour3Display.GaussianRadius = 3.5511111068725585
      contour3Display.SetScaleArray = ['POINTS', 'T']
      contour3Display.ScaleTransferFunction = 'PiecewiseFunction'
      contour3Display.OpacityArray = ['POINTS', 'T']
      contour3Display.OpacityTransferFunction = 'PiecewiseFunction'
      contour3Display.DataAxesGrid = 'GridAxesRepresentation'
      contour3Display.PolarAxes = 'PolarAxesRepresentation'

      # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
      contour3Display.ScaleTransferFunction.Points = [0.08454337722222222, 0.0, 0.5, 0.0, 0.7571025149999999, 1.0, 0.5, 0.0]

      # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
      contour3Display.OpacityTransferFunction.Points = [0.08454337722222222, 0.0, 0.5, 0.0, 0.7571025149999999, 1.0, 0.5, 0.0]

      # init the 'PolarAxesRepresentation' selected for 'PolarAxes'
      contour3Display.PolarAxes.Translation = [0.0, 0.0, -25.0]
      contour3Display.PolarAxes.Orientation = [-90.0, 0.0, 0.0]

      # show data from contour4
      contour4Display = Show(contour4, renderView1, 'GeometryRepresentation')

      # trace defaults for the display properties.
      contour4Display.Representation = 'Surface'
      contour4Display.ColorArrayName = ['POINTS', '']
      contour4Display.Position = [0.0, 0.0, -25.0]
      contour4Display.Orientation = [-90.0, 0.0, 0.0]
      contour4Display.OSPRayScaleArray = 'T'
      contour4Display.OSPRayScaleFunction = 'PiecewiseFunction'
      contour4Display.SelectOrientationVectors = 'U'
      contour4Display.ScaleFactor = 19.900000000000002
      contour4Display.SelectScaleArray = 'T'
      contour4Display.GlyphType = 'Arrow'
      contour4Display.GlyphTableIndexArray = 'T'
      contour4Display.GaussianRadius = 0.995
      contour4Display.SetScaleArray = ['POINTS', 'T']
      contour4Display.ScaleTransferFunction = 'PiecewiseFunction'
      contour4Display.OpacityArray = ['POINTS', 'T']
      contour4Display.OpacityTransferFunction = 'PiecewiseFunction'
      contour4Display.DataAxesGrid = 'GridAxesRepresentation'
      contour4Display.PolarAxes = 'PolarAxesRepresentation'

      # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
      contour4Display.ScaleTransferFunction.Points = [0.3787879943847656, 0.0, 0.5, 0.0, 0.3788490295410156, 1.0, 0.5, 0.0]

      # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
      contour4Display.OpacityTransferFunction.Points = [0.3787879943847656, 0.0, 0.5, 0.0, 0.3788490295410156, 1.0, 0.5, 0.0]

      # init the 'PolarAxesRepresentation' selected for 'PolarAxes'
      contour4Display.PolarAxes.Translation = [0.0, 0.0, -25.0]
      contour4Display.PolarAxes.Orientation = [-90.0, 0.0, 0.0]

      # show data from slice2
      slice2Display = Show(slice2, renderView1, 'GeometryRepresentation')

      # trace defaults for the display properties.
      slice2Display.Representation = 'Surface'
      slice2Display.ColorArrayName = ['POINTS', 'U']
      slice2Display.LookupTable = uLUT
      slice2Display.OSPRayScaleArray = 'T'
      slice2Display.OSPRayScaleFunction = 'PiecewiseFunction'
      slice2Display.SelectOrientationVectors = 'U'
      slice2Display.ScaleFactor = 79.9
      slice2Display.SelectScaleArray = 'T'
      slice2Display.GlyphType = 'Arrow'
      slice2Display.GlyphTableIndexArray = 'T'
      slice2Display.GaussianRadius = 3.995
      slice2Display.SetScaleArray = ['POINTS', 'T']
      slice2Display.ScaleTransferFunction = 'PiecewiseFunction'
      slice2Display.OpacityArray = ['POINTS', 'T']
      slice2Display.OpacityTransferFunction = 'PiecewiseFunction'
      slice2Display.DataAxesGrid = 'GridAxesRepresentation'
      slice2Display.PolarAxes = 'PolarAxesRepresentation'

      # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
      slice2Display.ScaleTransferFunction.Points = [0.000473485, 0.0, 0.5, 0.0, 0.757102515, 1.0, 0.5, 0.0]

      # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
      slice2Display.OpacityTransferFunction.Points = [0.000473485, 0.0, 0.5, 0.0, 0.757102515, 1.0, 0.5, 0.0]

      # setup the color legend parameters for each legend in this view

      # get color legend/bar for tLUT in view renderView1
      tLUTColorBar = GetScalarBar(tLUT, renderView1)
      tLUTColorBar.Orientation = 'Horizontal'
      tLUTColorBar.WindowLocation = 'AnyLocation'
      tLUTColorBar.Position = [0.575, 0.1]
      tLUTColorBar.Title = 'Temperature'
      tLUTColorBar.ComponentTitle = ''
      tLUTColorBar.TitleFontFamily = 'Courier'
      tLUTColorBar.TitleFontSize = 20
      tLUTColorBar.ScalarBarLength = 0.32999999999999985

      # set color bar visibility
      tLUTColorBar.Visibility = 1

      # get color legend/bar for uLUT in view renderView1
      uLUTColorBar = GetScalarBar(uLUT, renderView1)
      uLUTColorBar.Orientation = 'Horizontal'
      uLUTColorBar.WindowLocation = 'AnyLocation'
      uLUTColorBar.Position = [0.1785, 0.1]
      uLUTColorBar.Title = 'Lattice Velocity Magnitude'
      uLUTColorBar.ComponentTitle = 'Magnitude'
      uLUTColorBar.HorizontalTitle = 1
      uLUTColorBar.TitleFontFamily = 'Courier'
      uLUTColorBar.TitleFontSize = 20
      uLUTColorBar.ScalarBarLength = 0.32999999999999985

      # set color bar visibility
      uLUTColorBar.Visibility = 1

      # show color legend
      contour2Display.SetScalarBarVisibility(renderView1, True)

      # show color legend
      slice1Display.SetScalarBarVisibility(renderView1, True)

      # show color legend
      contour3Display.SetScalarBarVisibility(renderView1, True)

      # show color legend
      slice2Display.SetScalarBarVisibility(renderView1, True)

      # ----------------------------------------------------------------
      # setup color maps and opacity mapes used in the visualization
      # note: the Get..() functions create a new object, if needed
      # ----------------------------------------------------------------

      # get opacity transfer function/opacity map for 'T'
      tPWF = GetOpacityTransferFunction('T')
      tPWF.ScalarRangeInitialized = 1

      # get opacity transfer function/opacity map for 'U'
      uPWF = GetOpacityTransferFunction('U')
      uPWF.Points = [0.0, 0.0, 0.5, 0.0, 0.0005, 1.0, 0.5, 0.0]
      uPWF.ScalarRangeInitialized = 1

      # ----------------------------------------------------------------
      # finally, restore active source
      SetActiveSource(contour4)
      # ----------------------------------------------------------------
    return Pipeline()

  class CoProcessor(coprocessing.CoProcessor):
    def CreatePipeline(self, datadescription):
      self.Pipeline = _CreatePipeline(self, datadescription)

  coprocessor = CoProcessor()
  # these are the frequencies at which the coprocessor updates.
  freqs = {'input': [1]}
  coprocessor.SetUpdateFrequencies(freqs)
  if requestSpecificArrays:
    arrays = [['PhaseField', 0], ['ST', 0], ['T', 0], ['U', 0]]
    coprocessor.SetRequestedArrays('input', arrays)
  coprocessor.SetInitialOutputOptions(timeStepToStartOutputAt,forceOutputAtFirstCall)

  if rootDirectory:
      coprocessor.SetRootDirectory(rootDirectory)

  if make_cinema_table:
      coprocessor.EnableCinemaDTable()

  return coprocessor


#--------------------------------------------------------------
# Global variable that will hold the pipeline for each timestep
# Creating the CoProcessor object, doesn't actually create the ParaView pipeline.
# It will be automatically setup when coprocessor.UpdateProducers() is called the
# first time.
coprocessor = CreateCoProcessor()

#--------------------------------------------------------------
# Enable Live-Visualizaton with ParaView and the update frequency
coprocessor.EnableLiveVisualization(False, 1)

# ---------------------- Data Selection method ----------------------

def RequestDataDescription(datadescription):
    "Callback to populate the request for current timestep"
    global coprocessor

    # setup requests for all inputs based on the requirements of the
    # pipeline.
    coprocessor.LoadRequestedData(datadescription)

# ------------------------ Processing method ------------------------

def DoCoProcessing(datadescription):
    "Callback to do co-processing for current timestep"
    global coprocessor

    # Update the coprocessor by providing it the newly generated simulation data.
    # If the pipeline hasn't been setup yet, this will setup the pipeline.
    coprocessor.UpdateProducers(datadescription)

    # Write output data, if appropriate.
    coprocessor.WriteData(datadescription);

    # Write image capture (Last arg: rescale lookup table), if appropriate.
    coprocessor.WriteImages(datadescription, rescale_lookuptable=rescale_lookuptable,
        image_quality=0, padding_amount=imageFileNamePadding)

    # Live Visualization, if enabled.
    coprocessor.DoLiveVisualization(datadescription, "localhost", 22222)
