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
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def find_histogram(clt):
    """ create a histogram with k clusters
        :param: clt
        :return:hist
        """
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)
    
    hist = hist.astype("float")
    hist /= hist.sum()
    
    return hist
def plot_colors2(hist, centroids):
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0
    
    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX
    
    # return the bar chart
    return bar

#capture from FaceTime camera
cap = cv.VideoCapture(0)
plt.show()

# designated rectangle
w = 100
h = 100
x = 300
y = 300
color = (255, 0, 0)
thickness = 2


while True:
    ret, frame = cap.read()
    img = frame
    
    start_point = (x,y)
    end_point = (x+w, y+h)
    cropped_frame = frame[y:y+h, x:x+w]
    
    #draw a rectangle using cropped frame coord showing where the dominant color is going to be displayed
    img = cv.rectangle(img, start_point, end_point, color, thickness)
    cv.imshow('img',img)
    
    #cv.imshow('Cropped Image', cropped_frame)
    cropped_frame = cv.cvtColor(cropped_frame, cv.COLOR_BGR2RGB)
    cropped_frame = cropped_frame.reshape((cropped_frame.shape[0] * cropped_frame.shape[1],3)) #represent as row*column,channel number
    clt = KMeans(n_clusters=3) #cluster number
    clt.fit(cropped_frame)

    
    #bar plot
    hist = find_histogram(clt)
    bar = plot_colors2(hist, clt.cluster_centers_)

    plt.axis("off")
    plt.imshow(bar)
    plt.pause(0.001)

    k = cv.waitKey(1)
    if k == 27:
        break

cap.release()
cv.destroyAllWindows()

