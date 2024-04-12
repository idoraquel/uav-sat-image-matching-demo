import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

IMG_SRC = "uav"
# IMG_SRC = "sat"

for IMG_SRC in ["uav", "sat"]:

    img = cv.imread(f'data/test/{IMG_SRC}_images/DJI_0687.JPG', cv.IMREAD_GRAYSCALE) # `<opencv_root>/samples/data/blox.jpg`

    factor = 0.2
    interpolation = cv.INTER_LINEAR
    if IMG_SRC == "uav":
        interpolation = cv.INTER_NEAREST

    img = cv.resize(img, (0, 0), fx = factor, fy = factor, interpolation=interpolation)
    # ret, img = cv.threshold(img,127,230,cv.THRESH_TRUNC)
    # img = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,\
    #                            cv.THRESH_BINARY,11,2)
    # clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    # img = clahe.apply(img)
    # kernel1 = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
    # close = cv.morphologyEx(img,cv.MORPH_CLOSE,kernel1)
    # div = np.float32(img)/(close)
    # img = np.uint8(cv.normalize(div,div,0,255,cv.NORM_MINMAX))

    # Initiate FAST object with default values
    fast = cv.FastFeatureDetector_create()
    fast.setThreshold(20)
    fast.setType(1)

    cv.FastFeatureDetector

    # find and draw the keypoints
    kp = fast.detect(img, None)
    img2 = cv.drawKeypoints(img, kp, None, color=(255,0,0))
    
    # Print all default params
    print( "Threshold: {}".format(fast.getThreshold()) )
    print( "nonmaxSuppression:{}".format(fast.getNonmaxSuppression()) )
    print( "neighborhood: {}".format(fast.getType()) )
    print( "Total Keypoints with nonmaxSuppression: {}".format(len(kp)) )
    
    cv.imwrite(f'feature_extractors/fast/fast_true_{IMG_SRC}.png', img2)
