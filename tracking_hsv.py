#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# sources:
# https://answers.opencv.org/question/200861/drawing-a-rectangle-around-a-color-as-shown/
# https://code.likeagirl.io/finding-dominant-colour-on-an-image-b4e075f98097
# https://cvexplained.wordpress.com/2020/04/28/color-detection-hsv/
# https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_colorspaces/py_colorspaces.html

import cv2 as cv
import numpy as np

# parameter is device index 
#(ie the number specifying the camera)
#capture from FaceTime camera
cap = cv.VideoCapture(0)

# this is the blue we are looking for
bgr_blue = np.uint8([[[255, 51, 0]]])
hsv_blue = cv.cvtColor(bgr_blue,cv.COLOR_BGR2HSV)

while True:
    # if frame reads correctly, good returns True
    # frame is an numpy array of arrays
    good, frame = cap.read()
    
    # Convert BGR to HSV color space
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    
    # Define range of blue color in HSV
    lower_blue = np.array([hsv_blue[0][0][0] - 10, 50, 50]) #np.array([100,50,50])
    upper_blue =  np.array([hsv_blue[0][0][0] + 10, 255, 255]) #np.array([130,255,255])
    
    # Threshold the HSV image to get only blue colors
    # inRange(src, lower_bound_hsv, upper_bound_hsv)
    # In OpenCV, for HSV, Hue range is [0,179], Saturation range is [0,255] and Value range is [0,255]
    mask = cv.inRange (hsv, lower_blue, upper_blue)
    bluecnts = cv.findContours(mask.copy(),
                              cv.RETR_EXTERNAL,
                              cv.CHAIN_APPROX_SIMPLE)[-2]
    # For bounding box
    if len(bluecnts)>0:
        blue_area = max(bluecnts, key=cv.contourArea)
        (xg,yg,wg,hg) = cv.boundingRect(blue_area)
        cv.rectangle(frame,(xg,yg),(xg+wg, yg+hg),(0,255,0),2)

    cv.imshow('frame',frame)
    cv.imshow('mask',mask)

    k = cv.waitKey(5) 
    if k == 27:
        break

cap.release()
cv.destroyAllWindows()


# In[ ]:




