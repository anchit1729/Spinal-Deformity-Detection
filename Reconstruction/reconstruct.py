import sys
import cv2
import vtk
import math
import numpy as np

"""
AUTHOR: Anchit Mishra

This is a quick test program to see how OpenCV can visualise endplate coordinates 
given in a text file.
"""

def vector_norm(x, y, z):
    return math.sqrt(x**2 + y**2 + z**2)

def vector_dot(vec1, vec2):
    return vec1[0]*vec2[0] + vec1[1]*vec2[1] + vec1[2]*vec2[2]

# Name of the case (e.g. 01146, 01230, 00039 etc.)
instance_name = sys.argv[3]

img_src1 = f'./images/{instance_name}_AP.png'
img_src2 = f'./images/{instance_name}_LAT.png'
txt1 = f'./annotations/{instance_name}_AP.txt'
txt2 = f'./annotations/{instance_name}_LAT.txt'
pedicle_img = f'./images/{instance_name}_AP_prediction.png'

# Common width and height values are 448x896 and 696x892
width = int(sys.argv[1])
height = int(sys.argv[2])

# We also maintain a lookup table for w/(2d) values. 
# These are used by the Stokes method to estimate axial rotation angles.
ratio_lookup = {}
ratio_lookup['1'] = 1.24
ratio_lookup['2'] = 1.08
ratio_lookup['3'] = 0.97
ratio_lookup['4'] = 0.89
ratio_lookup['5'] = 0.83
ratio_lookup['6'] = 0.78
ratio_lookup['7'] = 0.75
ratio_lookup['8'] = 0.73
ratio_lookup['9'] = 0.71
ratio_lookup['10'] = 0.70
ratio_lookup['11'] = 0.69
ratio_lookup['12'] = 0.69

ratio_lookup['13'] = 0.69
ratio_lookup['14'] = 0.70
ratio_lookup['15'] = 0.71
ratio_lookup['16'] = 0.73
ratio_lookup['17'] = 0.79

pedicle_input = cv2.imread(pedicle_img)
pedicle_centroid_list = []

# Next, convert the image to grayscale 
gray_image = cv2.cvtColor(pedicle_input, cv2.COLOR_BGR2GRAY)
# convert the grayscale image to binary image
ret, thresh = cv2.threshold(gray_image, 127, 255, 0)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for c in contours:
    M = cv2.moments(c)
    x_coord = int(M['m10'] / M['m00'])
    y_coord = int(M['m01'] / M['m00'])
    pedicle_centroid_list.append((x_coord, y_coord))

pedicle_centroid_list = pedicle_centroid_list[2:]
pedicle_centroid_list.reverse()


# First, read the images
img1 = cv2.imread(img_src1)
img2 = cv2.imread(img_src2)
# Then, load in the coordinates
ap_endplate_coordinates = []
lat_endplate_coordinates = []
# AP coordinates are top-down, left to right
with open(txt1) as f:
    for line in f:
        line = line.split() # to deal with blank 
        if line:            # lines (ie skip them)
            line = [float(i) for i in line]
            ap_endplate_coordinates.append((int(line[0]*width), int(line[1]*height)))
# LAT coordinates are top-down, right to left (reason unknown, obtained from HKU Digital Health Lab)
with open(txt2) as f:
    for line in f:
        line = line.split() # to deal with blank 
        if line:            # lines (ie skip them)
            line = [float(i) for i in line]
            lat_endplate_coordinates.append((int(line[0]*width), int(line[1]*height)))
for i in range(len(lat_endplate_coordinates) // 2):
    temp = lat_endplate_coordinates[2*i]
    lat_endplate_coordinates[2*i] = lat_endplate_coordinates[2*i+1]
    lat_endplate_coordinates[2*i+1] = temp
# From experiments, it is known that the endplate coordinates are top-down, left-right
# Hence, we need to remove the last two coordinates
ap_endplate_coordinates = ap_endplate_coordinates[:-2]
lat_endplate_coordinates = lat_endplate_coordinates[:-2]
# And then take the bottom 17x4 = 68
ap_endplate_coordinates = ap_endplate_coordinates[-68:]
# Due to format errors in the endplate annotations, we need to take the bottom 69 
# Coordinates from the LAT view
lat_endplate_coordinates = lat_endplate_coordinates[-69:]

# ----------------------------------------------------------------------------- #
# This section is necessary for introducing Gaussian Noise into the dataset to observe resulting errors

# noise_factor = 1
# x_noise =  noise_factor * (width * 0.1)
# y_noise = noise_factor * (height * 0.1)

# for i in range(len(ap_endplate_coordinates)):
#     x = ap_endplate_coordinates[i][0]
#     y = ap_endplate_coordinates[i][1]
#     x += x_noise * np.random.normal()
#     y += y_noise * np.random.normal()
#     ap_endplate_coordinates[i] = (int(x), int(y))

# for i in range(len(lat_endplate_coordinates)):
#     x = lat_endplate_coordinates[i][0]
#     y = lat_endplate_coordinates[i][1]
#     x += x_noise * np.random.normal()
#     y += y_noise * np.random.normal()
#     lat_endplate_coordinates[i] = (int(x), int(y))

# ----------------------------------------------------------------------------- #

for i in range(len(ap_endplate_coordinates)):
    cv2.circle(img1, ap_endplate_coordinates[i], 3, (0, 255, 0), -1)
for i in range(len(lat_endplate_coordinates)):
    cv2.circle(img2, lat_endplate_coordinates[i], 3, (0, 255, 0), -1)
# We now obtain the center of each disc using average of all endplate coordinates
# Additionally, we compute the angles by averaging endplate vector angles
ap_disc_centers = [None for i in range(17)]
for i in range(17):
    x_coord = ap_endplate_coordinates[4*i][0] + ap_endplate_coordinates[4*i+1][0] + \
        ap_endplate_coordinates[4*i+2][0] + ap_endplate_coordinates[4*i+3][0]
    y_coord = ap_endplate_coordinates[4*i][1] + ap_endplate_coordinates[4*i+1][1] + \
        ap_endplate_coordinates[4*i+2][1] + ap_endplate_coordinates[4*i+3][1]
    ap_disc_centers[i] = (x_coord//4, y_coord//4)
    print(f'AP Disc Center #{i+1}: {ap_disc_centers[i]}')


lat_disc_centers = [None for i in range(17)]
for i in range(17):
    x_coord = lat_endplate_coordinates[4*i][0] + lat_endplate_coordinates[4*i+1][0] + \
        lat_endplate_coordinates[4*i+2][0] + lat_endplate_coordinates[4*i+3][0]
    y_coord = lat_endplate_coordinates[4*i][1] + lat_endplate_coordinates[4*i+1][1] + \
        lat_endplate_coordinates[4*i+2][1] + lat_endplate_coordinates[4*i+3][1]
    lat_disc_centers[i] = (x_coord//4, y_coord//4)
    print(f'LAT Disc Center #{i+1}: {lat_disc_centers[i]}')

# Now that AP disc centers have been computed and we also have access to the pedicle centers,
# We are able to compute the distance between pedicle centers and disc centers, and subsequently 
# The axial rotation angles.
rotation_angles_y = []
for i in range(17):
    delta_x_left = ap_disc_centers[i][0] - pedicle_centroid_list[2*i][0]
    delta_x_right = pedicle_centroid_list[2*i+1][0] - ap_disc_centers[i][0]
    w_d_ratio = ratio_lookup[str(i+1)]
    angle = math.atan(w_d_ratio * ((delta_x_right - delta_x_left)/(delta_x_right + delta_x_left)))
    angle *= (180.0 / math.pi)
    if width == 696 and height == 892:
        rotation_angles_y.append(angle)
    else:
        # Rotation calculations are only consistent on images of size
        # 696x892 since the pedicle segmentation maps are of the same size
        rotation_angles_y.append(0)
    #print(f'Delta Left: {delta_x_left}, Delta Right: {delta_x_right}')
    print(f'Rotation Angle {i+1}: {angle} degrees')

if width != 696 or height != 892:
    # Show a warning that axial rotation is not computed
    print('Due to image size inconsistencies, axial rotation is omitted!')

# For standardising the reference frame, we use the mid-point of the bottom-most 
# Endplate as our image origin as per the AP view. The doubt here is what if the
# LAT view origin doesn't align? For this, we keep the same point as origin for 
# Both images (to start off)
reference_point = (
    (ap_endplate_coordinates[-2][0] + ap_endplate_coordinates[-1][0])//2,
    (ap_endplate_coordinates[-2][1] + ap_endplate_coordinates[-1][1])//2
    )
cv2.circle(img1, reference_point, 3, (255, 0, 0), -1)
# In case we want to save the intermediate results
# cv2.imwrite(f'{img_src1[:-4]}_endplates.png', img1)
# cv2.imwrite(f'{img_src2[:-4]}_endplates.png', img2)
# Now, we move to vtk and try to load the image visualisations there
RENDER_IMAGE_AP = True
RENDER_IMAGE_LAT = True
RENDER_EXTRAPOLATE_AP = True
RENDER_EXTRAPOLATE_LAT = True
RENDER_VERTEBRAE = True

# We begin with the required readers
# First, for the thoracic vertebrae
readerThoracic = vtk.vtkSTLReader()
readerThoracic.SetFileName('human_thoracic_vertebra/model_keypoints.stl')
readerThoracic.Update()

# Then, for the lumbar vertebrae
readerLumbar = vtk.vtkSTLReader()
readerLumbar.SetFileName('human_lumbar_vertebra/model_keypoints.stl')
readerLumbar.Update()

# We also want a reader for processing images
# First, for the AP view
imageAPFileName = f'./images/{instance_name}_AP_endplates.png'
readerAP = vtk.vtkPNGReader()
readerAP.SetFileName(imageAPFileName)
readerAP.Update()
# Then, for the LAT view
imageLATFileName = f'./images/{instance_name}_LAT_endplates.png'
readerLAT = vtk.vtkPNGReader()
readerLAT.SetFileName(imageLATFileName)
readerLAT.Update()

# Next, we retrieve the image
imageLATData = readerLAT.GetOutput()
imageAPData = readerAP.GetOutput()
blue_point_world_coordinates = [0, 0, 0]
imageAPData.TransformContinuousIndexToPhysicalPoint(
    [reference_point[0], 896 - reference_point[1], 0],
    blue_point_world_coordinates
    )

# Retrieve endplate coordinates
endplate_list_world_ap = [None for i in range(68)]
for i in range(68):
    world_coords = [0, 0, 0]
    imageAPData.TransformContinuousIndexToPhysicalPoint(
        [ap_endplate_coordinates[i][0], 896 - ap_endplate_coordinates[i][1], 0],
        world_coords
        )
    coords_tuple = (world_coords[0], world_coords[1], world_coords[2])
    endplate_list_world_ap[i] = coords_tuple
ap_angles = [None for i in range(17)]
for i in range(17):
    upper_vector = (
        endplate_list_world_ap[4*i+1][0] - endplate_list_world_ap[4*i][0],
        endplate_list_world_ap[4*i+1][1] - endplate_list_world_ap[4*i][1],
        endplate_list_world_ap[4*i+1][2] - endplate_list_world_ap[4*i][2]
    )
    lower_vector = (
        endplate_list_world_ap[4*i+3][0] - endplate_list_world_ap[4*i+2][0],
        endplate_list_world_ap[4*i+3][1] - endplate_list_world_ap[4*i+2][1],
        endplate_list_world_ap[4*i+3][2] - endplate_list_world_ap[4*i+2][2]
    )
    angle_upper_vector = math.asin(
        ((upper_vector[1]) / vector_norm(upper_vector[0], upper_vector[1], upper_vector[2]))
    )
    angle_upper_vector *= (180.0 / math.pi)
    angle_lower_vector = math.asin(
        ((lower_vector[1]) / vector_norm(lower_vector[0], lower_vector[1], lower_vector[2]))
        ) 
    angle_lower_vector *= (180.0 / math.pi)
    ap_angles[i] = ((angle_lower_vector + angle_upper_vector) / 2)
    ap_angles[i] = angle_upper_vector
    if ap_angles[i] >= 180:
        ap_angles[i] -= 180

endplate_list_world_lat = [None for i in range(68)]
for i in range(68):
    world_coords = [0, 0, 0]
    imageLATData.TransformContinuousIndexToPhysicalPoint(
        [lat_endplate_coordinates[i][0], 896 - lat_endplate_coordinates[i][1], 0], 
        world_coords
        )
    coords_tuple = (world_coords[0], world_coords[1], world_coords[2])
    endplate_list_world_lat[i] = coords_tuple
lat_angles = [None for i in range(17)]
for i in range(17):
    upper_vector = (
        endplate_list_world_lat[4*i+1][0] - endplate_list_world_lat[4*i][0],
        endplate_list_world_lat[4*i+1][1] - endplate_list_world_lat[4*i][1],
        endplate_list_world_lat[4*i+1][2] - endplate_list_world_lat[4*i][2]
    )
    lower_vector = (
        endplate_list_world_lat[4*i+3][0] - endplate_list_world_lat[4*i+2][0],
        endplate_list_world_lat[4*i+3][1] - endplate_list_world_lat[4*i+2][1],
        endplate_list_world_lat[4*i+3][2] - endplate_list_world_lat[4*i+2][2]
    )
    angle_upper_vector = math.asin(
        ((upper_vector[1]) / vector_norm(upper_vector[0], upper_vector[1], upper_vector[2]))
    )
    angle_upper_vector *= (180.0 / math.pi)
    angle_lower_vector = math.asin(
        ((lower_vector[1]) / vector_norm(lower_vector[0], lower_vector[1], lower_vector[2]))
        ) 
    angle_lower_vector *= (180.0 / math.pi)
    lat_angles[i] = ((angle_lower_vector + angle_upper_vector) / 2)
    lat_angles[i] = angle_upper_vector
    if lat_angles[i] >= 180:
        lat_angles[i] -= 180

# Retrieve center coordinates
center_list_world_ap = [None for i in range(17)]
for i in range(17):
    world_coords = [0, 0, 0]
    imageAPData.TransformContinuousIndexToPhysicalPoint(
        [ap_disc_centers[i][0], 896 - ap_disc_centers[i][1], 0], 
        world_coords
        )
    coords_tuple = (world_coords[0], world_coords[1], world_coords[2])
    center_list_world_ap[i] = coords_tuple
    print(f'World coordinates of AP center #{i+1}: {center_list_world_ap[i]}')

center_list_world_lat = [None for i in range(17)]
for i in range(17):
    world_coords = [0, 0, 0]
    imageLATData.TransformContinuousIndexToPhysicalPoint(
        [lat_disc_centers[i][0], 896 - lat_disc_centers[i][1], 0], 
        world_coords
        )
    coords_tuple = (world_coords[0], world_coords[1], world_coords[2])
    center_list_world_lat[i] = coords_tuple
    print(f'World coordinates of LAT center #{i+1}: {center_list_world_lat[i]}')

# We would also like 3D coordinate axes to be plotted, so we define the points for that
origin = [0.0, 0.0, 0.0]
x_direction = [2000.0, 0.0, 0.0]
y_direction = [0.0, 2000.0, 0.0]
z_direction = [0.0, 0.0, 2000.0]

# Create line sources for the coordinate axes
lineSourceX = vtk.vtkLineSource()
lineSourceX.SetPoint1(origin)
lineSourceX.SetPoint2(x_direction)
lineSourceY = vtk.vtkLineSource()
lineSourceY.SetPoint1(origin)
lineSourceY.SetPoint2(y_direction)
lineSourceZ = vtk.vtkLineSource()
lineSourceZ.SetPoint1(origin)
lineSourceZ.SetPoint2(z_direction)


# Create line sources for the endplate lines
# 1: AP View
lineSourceListAP = [None for i in range(68)]
for i in range(68):
    lineSourceListAP[i] = vtk.vtkLineSource()
    lineSourceListAP[i].SetPoint1(
        [endplate_list_world_ap[i][0] - blue_point_world_coordinates[0], 
        endplate_list_world_ap[i][1] - blue_point_world_coordinates[1], 
        endplate_list_world_ap[i][2] - blue_point_world_coordinates[2] - 500]
        )
    lineSourceListAP[i].SetPoint2(
        [endplate_list_world_ap[i][0] - blue_point_world_coordinates[0], 
        endplate_list_world_ap[i][1] - blue_point_world_coordinates[1], 
        endplate_list_world_ap[i][2] - blue_point_world_coordinates[2] + 500]
        )

# 2: LAT View
lineSourceListLAT = [None for i in range(68)]
for i in range(68):
    lineSourceListLAT[i] = vtk.vtkLineSource()
    lineSourceListLAT[i].SetPoint1(
        [endplate_list_world_lat[i][0] - blue_point_world_coordinates[0], 
        endplate_list_world_lat[i][1] - blue_point_world_coordinates[1], 
        endplate_list_world_lat[i][2] - blue_point_world_coordinates[2] - 500]
        )
    lineSourceListLAT[i].SetPoint2(
        [endplate_list_world_lat[i][0] - blue_point_world_coordinates[0], 
        endplate_list_world_lat[i][1] - blue_point_world_coordinates[1], 
        endplate_list_world_lat[i][2] - blue_point_world_coordinates[2] + 500]
        )


# The pipeline then continues with mappers
# First, map the vertbrae
# 1: Thoracic Vertebrae
mapperSpineThoracic = vtk.vtkPolyDataMapper()
mapperSpineThoracic.SetInputConnection(readerThoracic.GetOutputPort())
# 2: Lumbar Vertebrae
mapperSpineLumbar = vtk.vtkPolyDataMapper()
mapperSpineLumbar.SetInputConnection(readerLumbar.GetOutputPort())

# Next, map the images
# 1: AP View
mapperAP = vtk.vtkImageMapToColors()
mapperAP.SetInputConnection(readerAP.GetOutputPort())

# 2: LAT View
mapperLAT = vtk.vtkImageMapToColors()
mapperLAT.SetInputConnection(readerLAT.GetOutputPort())

# Map the coordinate axes' lines
# 1: X-Axis
mapperXAxis = vtk.vtkPolyDataMapper()
mapperXAxis.SetInputConnection(lineSourceX.GetOutputPort())
# 2: Y-Axis
mapperYAxis = vtk.vtkPolyDataMapper()
mapperYAxis.SetInputConnection(lineSourceY.GetOutputPort())
# 3: Z-Axis
mapperZAxis = vtk.vtkPolyDataMapper()
mapperZAxis.SetInputConnection(lineSourceZ.GetOutputPort())

# Finally, map the endplate lines
# 1: AP View
mapperListAP = [None for i in range(68)]
for i in range(68):
    mapperListAP[i] = vtk.vtkPolyDataMapper()
    mapperListAP[i].SetInputConnection(lineSourceListAP[i].GetOutputPort())

# 2: LAT View
mapperListLAT = [None for i in range(68)]
for i in range(68):
    mapperListLAT[i] = vtk.vtkPolyDataMapper()
    mapperListLAT[i].SetInputConnection(lineSourceListLAT[i].GetOutputPort())

# Now, we create actors
# 12 Actors for the thoracic vertebrae
actorsSpineThoracic = [None for i in range(12)]
# 5 Actors for the lumbar vertebrae
actorsSpineLumbar = [None for i in range(5)]

# Assign mappers to the vertebrae
# 1: Assign mappers for thoracic vertebrae
for i in range(12):
    actorsSpineThoracic[i] = vtk.vtkActor()
    actorsSpineThoracic[i].SetMapper(mapperSpineThoracic)
# 2: Assign mappers for lumbar vertbrae
for i in range(5):
    actorsSpineLumbar[i] = vtk.vtkActor()
    actorsSpineLumbar[i].SetMapper(mapperSpineLumbar)

# Actor for the images
# 1: AP View
actorAP = vtk.vtkImageActor()
actorAP.SetInputData(imageAPData)

# 2: LAT View
actorLAT = vtk.vtkImageActor()
actorLAT.SetInputData(imageLATData)

actorXAxis = vtk.vtkActor()
actorXAxis.SetMapper(mapperXAxis)
actorXAxis.GetProperty().SetLineWidth(4)
actorXAxis.GetProperty().SetColor(0.0, 0.0, 1.0)
actorYAxis = vtk.vtkActor()
actorYAxis.SetMapper(mapperYAxis)
actorYAxis.GetProperty().SetLineWidth(4)
actorYAxis.GetProperty().SetColor(1.0, 1.0, 1.0)
actorZAxis = vtk.vtkActor()
actorZAxis.SetMapper(mapperZAxis)
actorZAxis.GetProperty().SetLineWidth(4)
actorZAxis.GetProperty().SetColor(1.0, 0.0, 0.0)

# Add the actors for the endplate lines
# 1: AP View
actorListAP = [None for i in range(68)]
for i in range(68):
    actorListAP[i] = vtk.vtkActor()
    actorListAP[i].SetMapper(mapperListAP[i])
    actorListAP[i].GetProperty().SetLineWidth(4)
    actorListAP[i].GetProperty().SetColor(0.0, 1.0, 0.0)

# 2: LAT View
actorListLAT = [None for i in range(68)]
for i in range(68):
    actorListLAT[i] = vtk.vtkActor()
    actorListLAT[i].SetMapper(mapperListLAT[i])
    actorListLAT[i].GetProperty().SetLineWidth(4)
    actorListLAT[i].GetProperty().SetColor(1.0, 0.0, 0.0)

# Finally, we set up the rendering and visualisation
window = vtk.vtkRenderWindow()
# We set the pixel width, length of the window.
window.SetSize(1500, 1500)

# The interactor enables the mouse interactions to view models
interactor = vtk.vtkRenderWindowInteractor()
# The trackball camera interactor style is the most comfortable to use
interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
# Make sure to set the renderer window!
interactor.SetRenderWindow(window)

# Next, create a renderer and attach it to the window for output
renderer = vtk.vtkRenderer()
window.AddRenderer(renderer)

# We add the actors to the renderer

# 1: Thoracic vertebrae actors
if RENDER_VERTEBRAE:
    for i in range(12):
        renderer.AddActor(actorsSpineThoracic[i])
# 2: Lumbar vertebrae actors
    for i in range(5):
        renderer.AddActor(actorsSpineLumbar[i])

# For testing out individual vertebrae
#renderer.AddActor(actorSpine)
renderer.AddActor(actorXAxis)
renderer.AddActor(actorYAxis)
renderer.AddActor(actorZAxis)
if RENDER_IMAGE_AP:
    renderer.AddActor(actorAP)
if RENDER_IMAGE_LAT:
    renderer.AddActor(actorLAT)
if RENDER_EXTRAPOLATE_AP:
    for i in range(68):
        renderer.AddActor(actorListAP[i])
if RENDER_EXTRAPOLATE_LAT:
    for i in range(68):
        renderer.AddActor(actorListLAT[i])

# We set the background to black.
renderer.SetBackground(0.0, 0.0, 0.0)
interactor.Initialize()
window.Render()

# Now, we perform some transforms in order to properly arrange the actor geometries in the model
# We need to set the x coordinate of the center list world lat as the z coordinate of the vertebrae as well
if RENDER_VERTEBRAE:
    # Since endplates go top-down, we first render the thoracic vertebrae
    for i in range(12):
        #actorsSpineThoracic[i].ForceTranslucentOn()
        actorsSpineThoracic[i].SetScale(0.9)
        actorsSpineThoracic[i].RotateZ(ap_angles[i])#40
        actorsSpineThoracic[i].RotateX(lat_angles[i])
        actorsSpineThoracic[i].RotateY(rotation_angles_y[i])
        print(f'Position of vertebra #{i+1}: {(center_list_world_ap[i][0] - blue_point_world_coordinates[0], ((center_list_world_ap[i][1] + center_list_world_lat[i][1]) / 2) - blue_point_world_coordinates[1], center_list_world_lat[i][0]+20-240)}')
        actorsSpineThoracic[i].SetPosition(
            center_list_world_ap[i][0] - blue_point_world_coordinates[0], 
            ((center_list_world_ap[i][1] + center_list_world_lat[i][1]) / 2) - blue_point_world_coordinates[1], 
            center_list_world_lat[i][0]+20-240
            )
        actorsSpineThoracic[i].SetOrigin(origin)
    # Then we render the lumbar vertebrae below the thoracic ones
    for i in range(12, 17):
        #actorsSpineLumbar[12 - i].ForceTranslucentOn()
        actorsSpineLumbar[12 - i].SetScale(0.9)
        actorsSpineLumbar[12 - i].RotateZ(ap_angles[i])#-65
        actorsSpineLumbar[12 - i].RotateX(lat_angles[i])
        actorsSpineLumbar[12 - i].RotateY(rotation_angles_y[i])
        print(f'Position of vertebra #{i+1}: {(center_list_world_ap[i][0] - blue_point_world_coordinates[0], ((center_list_world_ap[i][1] + center_list_world_lat[i][1]) / 2) - blue_point_world_coordinates[1], center_list_world_lat[i][0]-240)}')
        actorsSpineLumbar[12 - i].SetPosition(
            center_list_world_ap[i][0] - blue_point_world_coordinates[0], 
            ((center_list_world_ap[i][1] + center_list_world_lat[i][1]) / 2) - blue_point_world_coordinates[1], 
            center_list_world_lat[i][0]-240
            )
        actorsSpineLumbar[12 - i].SetOrigin(origin)

# We need to perform some transforms for the image as well
actorAP.SetScale(1)
actorAP.SetOrigin(origin)
actorAP.AddPosition(
    -1*blue_point_world_coordinates[0], 
    -1*blue_point_world_coordinates[1], 
    -1*blue_point_world_coordinates[2] - 240
    )

actorLAT.SetScale(1)
actorLAT.SetOrigin(origin)
actorLAT.AddPosition(
    -1*blue_point_world_coordinates[0], 
    -1*blue_point_world_coordinates[1], 
    -1*blue_point_world_coordinates[2] - 240
    )
actorLAT.RotateY(-90)
for i in range(68):
    actorListLAT[i].SetOrigin(origin)
    actorListLAT[i].RotateY(-90)
    if width == 696:
        actorListLAT[i].AddPosition(0, 0, 120)

window.Render()
interactor.Start()