import cv2
import numpy as np


#best is 00022.jpg
image = cv2.imread("../BCCD/JPEGImages/BloodImage_00023.jpg", 1)
image_bw = cv2.imread("../BCCD/JPEGImages/BloodImage_00001.jpg", 0)

blur = cv2.GaussianBlur(image[:,:,1], (5,5), 0)
_, otsu =cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

cv2.imshow("otsu", otsu)

kernel = np.ones((3,3),np.uint8)
#kernel = np.array([[0,1,0],[1,1,1],[0,1,0]], np.uint8)
erosion = cv2.erode(otsu,kernel,iterations = 5)

'''
cv2.imshow("orig", image)
cv2.imshow("test", erosion)
cv2.imwrite("test.jpg", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

#blue
blue = image.copy()
blue[:,:,1]=0
blue[:,:,2]=0
#cv2.imshow("blue", blue)

#green
green = image.copy()
green[:,:,0]=0
green[:,:,2]=0
#cv2.imshow("green", green)

#red
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

################WBC##############################################
kernel = np.ones((3,3),np.uint8)
thresh1 = cv2.dilate(thresh1,kernel,iterations = 5)

#thresh1[:,:,1]=0
#thresh1[:,:,2]=0
thresh1 = cv2.GaussianBlur(thresh1, (5,5), 0)

#extracting blue color
lower = np.array([0,0,0], dtype="uint8")
upper = np.array([255,0,0], dtype="uint8")
mask = cv2.inRange(thresh1, lower, upper)
output = cv2.bitwise_and(thresh1, thresh1, mask = mask)

kernel = np.ones((7,7),np.uint8)
output = cv2.erode(output ,kernel,iterations = 5)
output = cv2.dilate(output ,kernel,iterations = 5)
output = cv2.dilate(output ,kernel,iterations = 5)

#Blob
#ret,blob_bw = cv2.threshold(output,100,255,cv2.THRESH_TOZERO_INV)
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

keypoints_wbc = keypoints






cv2.imshow("images", np.hstack([output,image, wbc_blob]))


cv2.imshow("WBC", thresh1)
#cv2.imshow("WBC_blue", thresh1_blue)
#cv2.imshow("XWBC", image+thresh2)

cv2.imshow("original", image)
cv2.imshow("bw", image_bw)
#cv2.imshow("erosion", erosion)

#cv2.imshow("blue", blue)
#cv2.imshow("green", green)
#cv2.imshow("red", red)

#cv2.imshow("thres_bin_inv", thresh2)

'''cv2.imshow("thres_bin", thresh1)
cv2.imshow("thres_bin_inv", thresh2)
cv2.imshow("thres_bin_trunc", thresh3)
cv2.imshow("thres_bin_tozero", thresh4)
cv2.imshow("thres_bin_tozero_inv", thresh5)
'''








blur = cv2.GaussianBlur(image[:,:,1],(5,5),0)
ret3,green = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
ret3,blue = cv2.threshold(image[:,:,2],150,255,cv2.THRESH_BINARY_INV)
total = green + blue
kernel = np.ones((3,3), np.uint8)
img_erosion = cv2.erode(image[:,:,1], kernel, iterations=3)
eroded = cv2.erode(blue,kernel, iterations = 3)
opened = cv2.dilate(eroded,kernel,iterations= 3)
opened = cv2.subtract(opened, blob_thresh)
# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()
params.minThreshold = 10
params.maxThreshold = 255   
params.filterByArea = True
params.minArea = 50
params.maxArea = 1000    
params.filterByCircularity = False
params.filterByInertia = True
params.filterByConvexity = False
params.filterByColor = True
params.blobColor = 255

detector = cv2.SimpleBlobDetector_create(params)

keypoints = detector.detect(opened)
#f.write("{} platelets:{} wbc:{} \r\n".format(i,len(keypoints),len(keypoints_wbc)))
im_with_keypoints = cv2.drawKeypoints(image, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow("platelets", im_with_keypoints)




########################RBC#############################################
rbc0 = image.copy()
rbc_green = image.copy()
rbc_green[:,:,0]=0
rbc_green[:,:,2]=0
cv2.imshow("RBC_green",rbc_green[:,:,1])
ret,rbc1 = cv2.threshold(rbc_green,175,255,cv2.THRESH_TOZERO)

#kernel = np.ones((3,3),np.uint8)
kernel = np.array([[0,1,0],[1,1,1],[0,1,0]], np.uint8)
rbc2 = cv2.dilate(rbc1,kernel,iterations = 3)
rbc_edge = cv2.Canny(image_bw, 30, 30)
cv2.imshow("RBC_edge", rbc_edge)
cv2.imshow("RBC_erosion", rbc2)







#Blob
#ret,blob_bw = cv2.threshold(output,100,255,cv2.THRESH_TOZERO_INV)
blob_bw0 = cv2.cvtColor(rbc_green, cv2.COLOR_BGR2GRAY)
cv2.imshow("RBC_blob_bw0", blob_bw0)

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

kernel = np.array([[0,0,1,0,0],
[0,1,0,1,0],
[1,0,0,0,1],
[0,0,0,0,0],
[0,0,0,0,0]]
, np.uint8)

blob_bw = cv2.erode(blob_bw0,kernel,iterations = 1)

cv2.imshow("RBC_blob_bw", blob_bw)

ret, rbc1 = cv2.threshold(blob_bw,125,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow("RBC_blob_TOZERO", rbc1)

image_med = cv2.GaussianBlur(blob_bw,(3,3), 0.5)

blob_bw = rbc1
#_, blob_thresh =cv2.threshold(blob_bw,10 , 255, cv2.THRESH_BINARY)
#cv2.imshow("blob_thresh", blob_thresh)
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



#hough
rbc_green = image.copy()
rbc_green = cv2.subtract(rbc_green, wbc_blob)
ret,rbc_green = cv2.threshold(rbc_green,125,255,cv2.THRESH_TOZERO)
rbc_green[:,:,0]=0
rbc_green[:,:,2]=0
cv2.imshow("RBC_hough_green",rbc_green[:,:,1])

ret,rbc1 = cv2.threshold(rbc_green,175,255,cv2.THRESH_TOZERO)
kernel = np.ones((3,3),np.uint8)
#kernel = np.array([[0,1,0],[1,1,1],[0,1,0]], np.uint8)
rbc_green = cv2.erode(rbc_green,kernel,iterations = 5)
rbc_green = cv2.dilate(rbc_green,kernel,iterations = 5)
cv2.imshow("RBC_erosion", rbc_green)
image_med = cv2.GaussianBlur(rbc_green[:,:,1],(5,5), 0)

#cimg = cv2.cvtColor(image_med,cv2.COLOR_GRAY2BGR)
cimg = image.copy()

#hyperparam search 10,20,35,10
#b - probability threshold
#c - sensitivuty, how strong the edges of circles need ot be 
#d - how many edge points needs to be miniumn for classifying circle
# 2 100 50 45
a, b, c, d =(2,75,60,30)
#for a in np.random.randint(10, size=5):
#	print(a)


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
cv2.imshow("WBC_blob_hough", wbc_blob_hough)

ground = cv2.imread("../example.jpg", 1)
cv2.imwrite("result.jpg", np.hstack([wbc_blob_hough, ground]))

#cv2.imshow("RBC_hough", rbc_circle)




#segmentation
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


