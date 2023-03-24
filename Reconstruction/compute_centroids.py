import cv2

"""
AUTHOR: Anchit Mishra

This is a quick test program to see how OpenCV can implement calculation of pedicle 
centers and localisation of endplate coordinates, given the segmentation map 
generated by our deep learning model (UNet) and annotation .txt files for endplate coordinates.

"""

#img_src = './images/01230_AP_endplates.png'
img_src = './images/01230_AP_prediction.png'
img_display = './images/01230_LAT.png'

# first, read the image
input_image = cv2.imread(img_display)
pedicle_input = cv2.imread(img_src)
# input_image = cv2.resize(input_image, (696*2, 2*892))

pedicle_centroid_list = []

# Next, convert the image to grayscale 
gray_image = cv2.cvtColor(pedicle_input, cv2.COLOR_BGR2GRAY)
# convert the grayscale image to binary image
ret, thresh = cv2.threshold(gray_image, 127, 255, 0)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
for c in contours:
    M = cv2.moments(c)
    x_coord = int(M['m10'] / (M['m00']+1e-9))
    y_coord = int(M['m01'] / (M['m00']+1e-9))
    pedicle_centroid_list.append((x_coord, y_coord))

# sort the coordinates by increasing y coordinate
pedicle_centroid_list.sort(key=lambda x: (x[1], x[0]))

#pedicle_centroid_list = pedicle_centroid_list[2:]
#pedicle_centroid_list.reverse()

# # next, convert it to grayscale
# gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
# # convert the grayscale image to binary image
# ret, thresh = cv2.threshold(gray_image, 127, 255, 0)
# centroid_count = 0
# centroid_list = []

# # trace contours in the image
# contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# for c in contours:
#     M = cv2.moments(c)
#     x_coord = int(M['m10'] / M['m00'])
#     y_coord = int(M['m01'] / M['m00'])
#     centroid_list.append((x_coord, y_coord))
#     #cv2.circle(input_image, (x_coord, y_coord), 3, (255, 0, 0), -1)

# # Order of computing centroids is bottom-up, we only need the bottom 17 segmentations
# # Within same segmentation horizontally, order is left->right
# for i in range(17):
#     cv2.circle(input_image, centroid_list[2*i], 3, (0, 255, 0), -1)
#     cv2.circle(input_image, centroid_list[2*i+1], 3, (0, 255, 0), -1)
#     cv2.line(input_image, centroid_list[2*i], centroid_list[2*i+1], (0, 0, 255), thickness=1, lineType=8)
# cv2.circle(input_image, ((centroid_list[0][0] + centroid_list[1][0]) // 2, (centroid_list[0][1] + centroid_list[1][1]) // 2), 5, (255, 0, 0), -1)
txt1 = './annotations/01230_LAT.txt'
endplate_coordinates = []
with open(txt1) as f:
    for line in f:
        line = line.split() # to deal with blank 
        if line:            # lines (ie skip them)
            line = [float(i) for i in line]
            endplate_coordinates.append((int(line[0]*696), int(line[1]*892)))
            #endplate_coordinates.append((int(line[0]*448), int(line[1]*896)))
for i in range(len(endplate_coordinates) // 2):
    temp = endplate_coordinates[2*i]
    endplate_coordinates[2*i] = endplate_coordinates[2*i+1]
    endplate_coordinates[2*i+1] = temp
endplate_coordinates = endplate_coordinates[:-2]
endplate_coordinates = endplate_coordinates[-69:]
for i in range(len(endplate_coordinates) // 2):
    cv2.circle(input_image, endplate_coordinates[2*i], 3, (255, 0, 0), -1)
    cv2.circle(input_image, endplate_coordinates[2*i+1], 3, (255, 0, 0), -1)
    cv2.circle(input_image, pedicle_centroid_list[i], 3, (0, 255, 0), -1)
    cv2.line(input_image, endplate_coordinates[2*i], endplate_coordinates[2*i+1], (0, 0, 255), thickness=1, lineType=8)
    cv2.imshow('Endplates', input_image)
    cv2.waitKey(0)
cv2.imshow('Endplates', input_image)
cv2.waitKey(0)
input_image = cv2.resize(input_image, (696, 892))
cv2.imwrite(f'{img_src[:-4]}_centroids.png', input_image)