import cv2
import numpy as np


#best is 00022.jpg

#image: image of blood sample
#image_bw: black and white image of blood sample
image = cv2.imread("../BCCD/JPEGImages/BloodImage_00023.jpg", 1)
image_bw = cv2.imread("../BCCD/JPEGImages/BloodImage_00001.jpg", 0)


#Gaussian filter to remove noise
blur = cv2.GaussianBlur(image[:,:,1], (5,5), 0)
#Binarizing the image in black and white region based on intensity decided by otsu
_, otsu =cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

cv2.imshow("otsu", otsu)

#kernel: kernel for erosion 
#erosion used to remove or add boundaru to image objects
kernel = np.ones((3,3),np.uint8)
erosion = cv2.erode(otsu,kernel,iterations = 5)


#blue: blue channel
blue = image.copy()
blue[:,:,1]=0
blue[:,:,2]=0
#cv2.imshow("blue", blue)

#green: green channel
green = image.copy()
green[:,:,0]=0
green[:,:,2]=0
#cv2.imshow("green", green)

#red: red channel
red = image.copy()
red[:,:,0]=0
red[:,:,1]=0
#cv2.imshow("red", red)


#threshold
ret,thresh1 = cv2.threshold(image,200,255,cv2.THRESH_BINARY)
ret,thresh2 = cv2.threshold(image,127,255,cv2.THRESH_BINARY_INV)
ret,thresh3 = cv2.threshold(image,127,255,cv2.THRESH_TRUNC)
ret,thresh4 = cv2.threshold(image,127,255,cv2.THRESH_TOZERO)
ret,thresh5 = cv2.threshold(image,127,255,cv2.THRESH_TOZERO_INV)

thresh1_blue = thresh1.copy()
thresh1_blue[:,:,0]=0
thresh1_blue[:,:,2]=0


################ WBC Detection ##############################################
#Dilation
kernel = np.ones((3,3),np.uint8)
thresh1 = cv2.dilate(thresh1,kernel,iterations = 5)
#Gaussian Blur
thresh1 = cv2.GaussianBlur(thresh1, (5,5), 0)

#extracting blue color
lower = np.array([0,0,0], dtype="uint8")
upper = np.array([255,0,0], dtype="uint8")
mask = cv2.inRange(thresh1, lower, upper)
output = cv2.bitwise_and(thresh1, thresh1, mask = mask)

#erosion and dilation to seperate the boundaries of WBC and filling the sparse blob of WBC
kernel = np.ones((7,7),np.uint8)
output = cv2.erode(output ,kernel,iterations = 5)
output = cv2.dilate(output ,kernel,iterations = 5)
output = cv2.dilate(output ,kernel,iterations = 5)

##Blob detection of the WBC patches
blob_bw = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
_, blob_thresh =cv2.threshold(blob_bw,10 , 255, cv2.THRESH_BINARY)
cv2.imshow("blob_thresh", blob_thresh)
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 0
params.maxThreshold = 255
# Filter by Area.
params.filterByArea = False
params.minArea = 0
params.maxArea = 10000
# Filter by Circularity
params.filterByCircularity = False
#params.minCircularity = 0.1
# Filter by Convexity
params.filterByConvexity = False
#params.minConvexity = 0.87
# Filter by Inertia
params.filterByInertia = False
#params.minInertiaRatio = 0.01
params.filterByColor = False
params.blobColor = 255
# Create a detector with the parameters
ver = (cv2.__version__).split('.')
if int(ver[0]) < 3 :
	detector = cv2.SimpleBlobDetector(params)
else : 
	detector = cv2.SimpleBlobDetector_create(params)
# Detect blobs.
keypoints = detector.detect(blob_thresh)

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
# the size of the circle corresponds to the size of blob

wbc_blob = cv2.drawKeypoints(blob_thresh, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show blobs
cv2.imshow("WBC_blob", wbc_blob)
#save keypoints found in wbc patches
keypoints_wbc = keypoints

#Display eroded and dilated WBC patch, orginal image and wbc blob as a stack in single window
cv2.imshow("images", np.hstack([output,image, wbc_blob]))


cv2.imshow("WBC", thresh1)
cv2.imshow("original", image)
cv2.imshow("bw", image_bw)


######################## RBC Detection #############################################

#Combining the candidates chosen from Blob and Hough Transform and will give us better results after non-max supression. Combining and Non-Max Supression is not implemented here.

rbc0 = image.copy()

#green channel
rbc_green = image.copy()
rbc_green[:,:,0]=0
rbc_green[:,:,2]=0
cv2.imshow("RBC_green",rbc_green[:,:,1])
#thresholding
ret,rbc1 = cv2.threshold(rbc_green,175,255,cv2.THRESH_TOZERO)

#dilation
kernel = np.array([[0,1,0],[1,1,1],[0,1,0]], np.uint8)
rbc2 = cv2.dilate(rbc1,kernel,iterations = 3)
#edge detection: not helpful 
rbc_edge = cv2.Canny(image_bw, 30, 30)
cv2.imshow("RBC_edge", rbc_edge)
cv2.imshow("RBC_erosion", rbc2)



##Blob detection of RBC 
blob_bw0 = cv2.cvtColor(rbc_green, cv2.COLOR_BGR2GRAY)
cv2.imshow("RBC_blob_bw0", blob_bw0)

#erosion: not helpful
kernel = np.ones((3,3),np.uint8)
#kernel = np.array([[0,0,1,0,0],
#[0,1,1,1,0],
#[1,1,1,1,1],
#[0,1,1,1,0],
#[0,0,1,0,0]]
#, np.uint8)

#kernel = np.array([[0,0,1,0,0],
#[0,1,1,1,0],
#[1,1,1,1,1],
#[0,1,1,1,0],
#[0,0,1,0,0]], np.uint8)

#though that maybe a semicircular kernel may help in making the boundaries of rbc prominent
kernel = np.array([[0,0,1,0,0],
[0,1,0,1,0],
[1,0,0,0,1],
[0,0,0,0,0],
[0,0,0,0,0]]
, np.uint8)
#eroded black and white image
blob_bw = cv2.erode(blob_bw0,kernel,iterations = 1)
cv2.imshow("RBC_blob_bw", blob_bw)

#otsu binarization
ret, rbc1 = cv2.threshold(blob_bw,125,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow("RBC_blob_TOZERO", rbc1)

#Gaussian to remove edges a bit from the cytoplasm so that they are not detected
image_med = cv2.GaussianBlur(blob_bw,(3,3), 0.5)
blob_bw = rbc1

params = cv2.SimpleBlobDetector_Params()
# Change thresholds
params.minThreshold = 0
params.maxThreshold = 255
# Filter by Area.
params.filterByArea = True
params.minArea = 0
params.maxArea = 2000
# Filter by Circularity
params.filterByCircularity = False
#params.minCircularity = 0.1
# Filter by Convexity
params.filterByConvexity = False
#params.minConvexity = 0.87
# Filter by Inertia
params.filterByInertia = False
#params.minInertiaRatio = 0.01
params.filterByColor = True
params.blobColor = 255
# Create a detector with the parameters
ver = (cv2.__version__).split('.')
if int(ver[0]) < 3 :
	detector = cv2.SimpleBlobDetector(params)
else : 
	detector = cv2.SimpleBlobDetector_create(params)
# Detect blobs.
#keypoints = detector.detect(blob_thresh)
keypoints = detector.detect(blob_bw)

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
# the size of the circle corresponds to the size of blob

rbc_blob = cv2.drawKeypoints(image, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show blobs
cv2.imshow("RBC_blob", rbc_blob)
#cv2.imshow("RBC_images", np.hstack([image, rbc_blob, rbc_circle]))



##Hough Circle Transform
rbc_green = image.copy()
#subtract the WBC from image so that they are detected while detected RBC's
rbc_green = cv2.subtract(rbc_green, wbc_blob)
#Thresholding
ret,rbc_green = cv2.threshold(rbc_green,125,255,cv2.THRESH_TOZERO)
#green channel as the image in this case more prominent in green channel
rbc_green[:,:,0]=0
rbc_green[:,:,2]=0
cv2.imshow("RBC_hough_green",rbc_green[:,:,1])

#Thresholding
ret,rbc1 = cv2.threshold(rbc_green,175,255,cv2.THRESH_TOZERO)
#Erosion and dilation to make the boundaries of RBC more prominent
kernel = np.ones((3,3),np.uint8)
rbc_green = cv2.erode(rbc_green,kernel,iterations = 5)
rbc_green = cv2.dilate(rbc_green,kernel,iterations = 5)
cv2.imshow("RBC_erosion", rbc_green)

#Gaussian filter to prevent the edges in cytoplasm from fooling the algo to detecting it as circle
image_med = cv2.GaussianBlur(rbc_green[:,:,1],(5,5), 0)
#cimg = cv2.cvtColor(image_med,cv2.COLOR_GRAY2BGR)
cimg = image.copy()

#Hyperparam search for Hough Transform (10,20,35,10)
#b - probability threshold, higher meand that the circles which have more probability are showed
#c(param1) - sensitivuty, how strong the edges of circles need to be 
#d(param2) - how many edge points needs to be miniumn for classifying circle
# 2 100 50 45

a, b, c, d =(2,75,60,30)

circles = cv2.HoughCircles(image_med,cv2.HOUGH_GRADIENT,a,b,  param1=c,param2=d,minRadius=30,maxRadius=60)
circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

rbc_circle = cimg
#cv2.imwrite("hough/RBC_circles"+str(a)+" "+str(b)+" "+str(c)+" "+str(d)+".jpg",rbc_circle)

wbc_blob_hough = cv2.drawKeypoints(rbc_circle, keypoints_wbc, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show blobs
#cv2.imshow("RBC_hough", rbc_circle)


#Show circle from Blobs and Hough Transform  
cv2.imshow("WBC_blob_hough", wbc_blob_hough)



#Segmentation using Watershed and Distance Transform: not helpful
ret, thresh = cv2.threshold(image_bw,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
# sure background area
sure_bg = cv2.dilate(opening,kernel,iterations=3)
# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.2*dist_transform.max(),255,0)
# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)
# Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)
# Add one to all labels so that sure background is not 0, but 1
markers = markers+1
# Now, mark the region of unknown with zero
markers[unknown==255] = 0
markers = cv2.watershed(image,markers)
image[markers == -1] = [255,0,0]

#cv2.imshow("segemnt", image)

#cv2.imshow("RBC0", rbc0)
#cv2.imshow("RBC1", rbc1)	
#cv2.imshow("RBC_edge", rbc_edge)




cv2.waitKey(0)
cv2.destroyAllWindows()


