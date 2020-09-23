
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
      renderView1.ViewSize = [1504, 796]
      renderView1.InteractionMode = '2D'
      renderView1.AxesGrid = 'GridAxes3DActor'
      renderView1.CenterOfRotation = [19.91768115443716, -13.72628195506752, 100.00000000000003]
      renderView1.StereoType = 'Crystal Eyes'
      renderView1.CameraPosition = [372.17218180684347, 162.99002272005248, 1704.7165036923268]
      renderView1.CameraFocalPoint = [372.17218180684347, 162.99002272005248, 100.0]
      renderView1.CameraFocalDisk = 1.0
      renderView1.CameraParallelScale = 234.44363086899264
      renderView1.BackEnd = 'OSPRay raycaster'
      renderView1.OSPRayMaterialLibrary = materialLibrary1

      # register the view with coprocessor
      # and provide it with information such as the filename to use,
      # how frequently to write the images, etc.
      coprocessor.RegisterView(renderView1,
          filename='RenderView1_%t.png', freq=1, fittoscreen=0, magnification=1, width=1504, height=796, cinema={}, compression=5)
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

      # create a new 'XML Partitioned Image Data Reader'
      # create a producer from a simulation input
      recalcitrantBubble_mm_200623_250_actual_VTK_P00_00000600pvti = coprocessor.CreateProducer(datadescription, 'input')

      # create a new 'Cell Data to Point Data'
      cellDatatoPointData1 = CellDatatoPointData(Input=recalcitrantBubble_mm_200623_250_actual_VTK_P00_00000600pvti)
      cellDatatoPointData1.CellDataArraytoprocess = ['ADDITIONALS', 'BOUNDARY', 'COLLISION', 'DESIGNSPACE', 'NONE', 'Normal', 'P', 'PhaseField', 'Rho', 'SETTINGZONE', 'ST', 'T', 'U', 'flag']

      # create a new 'Slice'
      slice1 = Slice(Input=cellDatatoPointData1)
      slice1.SliceType = 'Plane'
      slice1.HyperTreeGridSlicer = 'Plane'
      slice1.SliceOffsetValues = [0.0]

      # init the 'Plane' selected for 'SliceType'
      slice1.SliceType.Origin = [400.0, 50.0, 100.0]
      slice1.SliceType.Normal = [0.0, 1.0, 0.0]

      # init the 'Plane' selected for 'HyperTreeGridSlicer'
      slice1.HyperTreeGridSlicer.Origin = [400.0, 50.0, 100.0]

      # create a new 'Contour'
      contour2 = Contour(Input=slice1)
      contour2.ContourBy = ['POINTS', 'T']
      contour2.Isosurfaces = [0.000473485, 0.040296065526315795, 0.08011864605263158, 0.11994122657894737, 0.15976380710526317, 0.19958638763157896, 0.23940896815789475, 0.2792315486842105, 0.31905412921052634, 0.35887670973684216, 0.3986992902631579, 0.4385218707894737, 0.4783444513157895, 0.5181670318421053, 0.557989612368421, 0.5978121928947369, 0.6376347734210527, 0.6774573539473685, 0.7172799344736843, 0.757102515]
      contour2.PointMergeMethod = 'Uniform Binning'

      # create a new 'Contour'
      contour3 = Contour(Input=cellDatatoPointData1)
      contour3.ContourBy = ['POINTS', 'PhaseField']
      contour3.Isosurfaces = [0.5]
      contour3.PointMergeMethod = 'Uniform Binning'

      # create a new 'Contour'
      contour1 = Contour(Input=cellDatatoPointData1)
      contour1.ContourBy = ['POINTS', 'PhaseField']
      contour1.Isosurfaces = [0.5]
      contour1.PointMergeMethod = 'Uniform Binning'

      # ----------------------------------------------------------------
      # setup the visualization in view 'renderView1'
      # ----------------------------------------------------------------

      # show data from cellDatatoPointData1
      cellDatatoPointData1Display = Show(cellDatatoPointData1, renderView1, 'UniformGridRepresentation')

      # trace defaults for the display properties.
      cellDatatoPointData1Display.Representation = 'Outline'
      cellDatatoPointData1Display.ColorArrayName = [None, '']
      cellDatatoPointData1Display.OSPRayScaleArray = 'ADDITIONALS'
      cellDatatoPointData1Display.OSPRayScaleFunction = 'PiecewiseFunction'
      cellDatatoPointData1Display.SelectOrientationVectors = 'ADDITIONALS'
      cellDatatoPointData1Display.ScaleFactor = 80.0
      cellDatatoPointData1Display.SelectScaleArray = 'ADDITIONALS'
      cellDatatoPointData1Display.GlyphType = 'Arrow'
      cellDatatoPointData1Display.GlyphTableIndexArray = 'ADDITIONALS'
      cellDatatoPointData1Display.GaussianRadius = 4.0
      cellDatatoPointData1Display.SetScaleArray = ['POINTS', 'ADDITIONALS']
      cellDatatoPointData1Display.ScaleTransferFunction = 'PiecewiseFunction'
      cellDatatoPointData1Display.OpacityArray = ['POINTS', 'ADDITIONALS']
      cellDatatoPointData1Display.OpacityTransferFunction = 'PiecewiseFunction'
      cellDatatoPointData1Display.DataAxesGrid = 'GridAxesRepresentation'
      cellDatatoPointData1Display.PolarAxes = 'PolarAxesRepresentation'
      cellDatatoPointData1Display.ScalarOpacityUnitDistance = 3.296485864575075
      cellDatatoPointData1Display.SliceFunction = 'Plane'
      cellDatatoPointData1Display.Slice = 100

      # init the 'Plane' selected for 'SliceFunction'
      cellDatatoPointData1Display.SliceFunction.Origin = [400.0, 50.0, 100.0]

      # show data from contour1
      contour1Display = Show(contour1, renderView1, 'GeometryRepresentation')

      # get color transfer function/color map for 'PhaseField'
      phaseFieldLUT = GetColorTransferFunction('PhaseField')
      phaseFieldLUT.RGBPoints = [0.5, 0.231373, 0.298039, 0.752941, 0.50006103515625, 0.865003, 0.865003, 0.865003, 0.5001220703125, 0.705882, 0.0156863, 0.14902]
      phaseFieldLUT.ScalarRangeInitialized = 1.0

      # trace defaults for the display properties.
      contour1Display.Representation = 'Surface'
      contour1Display.ColorArrayName = ['POINTS', 'PhaseField']
      contour1Display.LookupTable = phaseFieldLUT
      contour1Display.OSPRayScaleArray = 'PhaseField'
      contour1Display.OSPRayScaleFunction = 'PiecewiseFunction'
      contour1Display.SelectOrientationVectors = 'ADDITIONALS'
      contour1Display.ScaleFactor = 4.997750854492188
      contour1Display.SelectScaleArray = 'PhaseField'
      contour1Display.GlyphType = 'Arrow'
      contour1Display.GlyphTableIndexArray = 'PhaseField'
      contour1Display.GaussianRadius = 0.2498875427246094
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

      # show data from slice1
      slice1Display = Show(slice1, renderView1, 'GeometryRepresentation')

      # get color transfer function/color map for 'T'
      tLUT = GetColorTransferFunction('T')
      tLUT.RGBPoints = [0.000473485, 0.0, 0.0, 0.0, 0.303125097, 0.901960784314, 0.0, 0.0, 0.605776709, 0.901960784314, 0.901960784314, 0.0, 0.757102515, 1.0, 1.0, 1.0]
      tLUT.ColorSpace = 'RGB'
      tLUT.NanColor = [0.0, 0.498039215686, 1.0]
      tLUT.ScalarRangeInitialized = 1.0

      # trace defaults for the display properties.
      slice1Display.Representation = 'Surface'
      slice1Display.ColorArrayName = ['POINTS', 'T']
      slice1Display.LookupTable = tLUT
      slice1Display.Position = [0.0, 350.0, 0.0]
      slice1Display.Orientation = [90.0, 0.0, 0.0]
      slice1Display.OSPRayScaleArray = 'ADDITIONALS'
      slice1Display.OSPRayScaleFunction = 'PiecewiseFunction'
      slice1Display.SelectOrientationVectors = 'ADDITIONALS'
      slice1Display.ScaleFactor = 80.0
      slice1Display.SelectScaleArray = 'ADDITIONALS'
      slice1Display.GlyphType = 'Arrow'
      slice1Display.GlyphTableIndexArray = 'ADDITIONALS'
      slice1Display.GaussianRadius = 4.0
      slice1Display.SetScaleArray = ['POINTS', 'ADDITIONALS']
      slice1Display.ScaleTransferFunction = 'PiecewiseFunction'
      slice1Display.OpacityArray = ['POINTS', 'ADDITIONALS']
      slice1Display.OpacityTransferFunction = 'PiecewiseFunction'
      slice1Display.DataAxesGrid = 'GridAxesRepresentation'
      slice1Display.PolarAxes = 'PolarAxesRepresentation'

      # init the 'PolarAxesRepresentation' selected for 'PolarAxes'
      slice1Display.PolarAxes.Translation = [0.0, 350.0, 0.0]
      slice1Display.PolarAxes.Orientation = [90.0, 0.0, 0.0]

      # show data from contour2
      contour2Display = Show(contour2, renderView1, 'GeometryRepresentation')

      # trace defaults for the display properties.
      contour2Display.Representation = 'Surface'
      contour2Display.ColorArrayName = ['POINTS', '']
      contour2Display.DiffuseColor = [0.0, 0.0, 0.0]
      contour2Display.Position = [0.0, 350.0, 0.0]
      contour2Display.Orientation = [90.0, 0.0, 0.0]
      contour2Display.OSPRayScaleArray = 'T'
      contour2Display.OSPRayScaleFunction = 'PiecewiseFunction'
      contour2Display.SelectOrientationVectors = 'ADDITIONALS'
      contour2Display.ScaleFactor = 75.73573837280274
      contour2Display.SelectScaleArray = 'T'
      contour2Display.GlyphType = 'Arrow'
      contour2Display.GlyphTableIndexArray = 'T'
      contour2Display.GaussianRadius = 3.7867869186401366
      contour2Display.SetScaleArray = ['POINTS', 'T']
      contour2Display.ScaleTransferFunction = 'PiecewiseFunction'
      contour2Display.OpacityArray = ['POINTS', 'T']
      contour2Display.OpacityTransferFunction = 'PiecewiseFunction'
      contour2Display.DataAxesGrid = 'GridAxesRepresentation'
      contour2Display.PolarAxes = 'PolarAxesRepresentation'

      # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
      contour2Display.ScaleTransferFunction.Points = [0.040296065526315795, 0.0, 0.5, 0.0, 0.757102515, 1.0, 0.5, 0.0]

      # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
      contour2Display.OpacityTransferFunction.Points = [0.040296065526315795, 0.0, 0.5, 0.0, 0.757102515, 1.0, 0.5, 0.0]

      # init the 'PolarAxesRepresentation' selected for 'PolarAxes'
      contour2Display.PolarAxes.Translation = [0.0, 350.0, 0.0]
      contour2Display.PolarAxes.Orientation = [90.0, 0.0, 0.0]

      # show data from contour3
      contour3Display = Show(contour3, renderView1, 'GeometryRepresentation')

      # trace defaults for the display properties.
      contour3Display.Representation = 'Surface'
      contour3Display.ColorArrayName = ['POINTS', 'PhaseField']
      contour3Display.LookupTable = phaseFieldLUT
      contour3Display.Position = [0.0, 350.0, 0.0]
      contour3Display.Orientation = [90.0, 0.0, 0.0]
      contour3Display.OSPRayScaleArray = 'PhaseField'
      contour3Display.OSPRayScaleFunction = 'PiecewiseFunction'
      contour3Display.SelectOrientationVectors = 'ADDITIONALS'
      contour3Display.ScaleFactor = 4.997750854492188
      contour3Display.SelectScaleArray = 'PhaseField'
      contour3Display.GlyphType = 'Arrow'
      contour3Display.GlyphTableIndexArray = 'PhaseField'
      contour3Display.GaussianRadius = 0.2498875427246094
      contour3Display.SetScaleArray = ['POINTS', 'PhaseField']
      contour3Display.ScaleTransferFunction = 'PiecewiseFunction'
      contour3Display.OpacityArray = ['POINTS', 'PhaseField']
      contour3Display.OpacityTransferFunction = 'PiecewiseFunction'
      contour3Display.DataAxesGrid = 'GridAxesRepresentation'
      contour3Display.PolarAxes = 'PolarAxesRepresentation'

      # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
      contour3Display.ScaleTransferFunction.Points = [0.5, 0.0, 0.5, 0.0, 0.5001220703125, 1.0, 0.5, 0.0]

      # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
      contour3Display.OpacityTransferFunction.Points = [0.5, 0.0, 0.5, 0.0, 0.5001220703125, 1.0, 0.5, 0.0]

      # init the 'PolarAxesRepresentation' selected for 'PolarAxes'
      contour3Display.PolarAxes.Translation = [0.0, 350.0, 0.0]
      contour3Display.PolarAxes.Orientation = [90.0, 0.0, 0.0]

      # ----------------------------------------------------------------
      # setup color maps and opacity mapes used in the visualization
      # note: the Get..() functions create a new object, if needed
      # ----------------------------------------------------------------

      # get opacity transfer function/opacity map for 'PhaseField'
      phaseFieldPWF = GetOpacityTransferFunction('PhaseField')
      phaseFieldPWF.Points = [0.5, 0.0, 0.5, 0.0, 0.5001220703125, 1.0, 0.5, 0.0]
      phaseFieldPWF.ScalarRangeInitialized = 1

      # get opacity transfer function/opacity map for 'T'
      tPWF = GetOpacityTransferFunction('T')
      tPWF.Points = [0.000473485, 0.0, 0.5, 0.0, 0.757102515, 1.0, 0.5, 0.0]
      tPWF.ScalarRangeInitialized = 1

      # ----------------------------------------------------------------
      # finally, restore active source
      SetActiveSource(contour3)
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
    arrays = [['ADDITIONALS', 1], ['BOUNDARY', 1], ['COLLISION', 1], ['DESIGNSPACE', 1], ['flag', 1], ['NONE', 1], ['Normal', 1], ['P', 1], ['PhaseField', 1], ['Rho', 1], ['SETTINGZONE', 1], ['ST', 1], ['T', 1], ['U', 1]]
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
