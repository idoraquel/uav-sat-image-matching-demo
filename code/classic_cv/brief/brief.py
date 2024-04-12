import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

IMG_SRC = "uav"
# IMG_SRC = "sat"

kp = list()
des = list()
imgs = []

for IMG_SRC in ["uav", "sat"]:

    img = cv.imread(f'data/test/{IMG_SRC}_images/DJI_0407.JPG', cv.IMREAD_GRAYSCALE) # `<opencv_root>/samples/data/blox.jpg`

    factor = 0.1
    if IMG_SRC == "sat":
        factor = 0.1

    img = cv.resize(img, (0, 0), fx = factor, fy = factor)
    imgs.append(img)
    
    fast = cv.FastFeatureDetector_create()
    fast.setThreshold(20)
    
    # Initiate BRIEF extractor
    brief = cv.xfeatures2d.BriefDescriptorExtractor_create()
    
    # find the keypoints with STAR
    kp_ = fast.detect(img,None)
    
    # compute the descriptors with BRIEF
    kp_, des_ = brief.compute(img, kp_)

    kp.append(kp_), des.append(des_)

    img2 = cv.drawKeypoints(img, kp_, None, color=(255,0,0))
    
    cv.imwrite(f'feature_extractors/brief/brief_true_{IMG_SRC}.png', img2)

# create BFMatcher object
bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des[0],des[1])

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 10 matches.
img3 = cv.drawMatches(imgs[0],kp[0],imgs[1],kp[1],matches[:20],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.imshow(img3),plt.show()
