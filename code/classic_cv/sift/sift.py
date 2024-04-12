import numpy as np
import cv2 as cv
 
img = cv.imread('data/test/sat_images/DJI_0267.JPG')
gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)

gray = cv.resize(gray, (0, 0), fx = 0.5, fy = 0.5)

sift = cv.SIFT_create()
kp = sift.detect(gray,None)

img=cv.drawKeypoints(gray,kp,img,color=(0, 255, 0),flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv.imwrite('sift_keypoints_half.jpg',img)