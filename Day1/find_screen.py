#import the necessary packages
import imutils
from skimage import exposure
import numpy as np
import cv2

#load the image
image = cv2.imread('image.jpg')
ratio = image.shape[0] / 300.0 #height
orig = image.copy()
image = imutils.resize(image,height = 300)

#convert the image to grayscale,blur it,find edges
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray,11,17,17)
edged = cv2.Canny(gray,30,200)

#find contours in the edged image,keep
#only the largest
cnts = cv2.findContours(edged.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts,key = cv2.contourArea,reverse = True)[:10]
screenCnt = None

#loop over our contours
for c in cnts:
    #approximate the contour
    peri = cv2.arcLength(c,True)
    approx = cv2.approxPolyDP(c,0.015 * peri,True)
    #if our approximated contour has four points,then
    #we can assume that we have found our screen
    if len(approx) == 4:
        screenCnt = approx
        break
cv2.drawContours(image,[screenCnt],-1,(0,255,0),3)
cv2.imwrite('Game.jpg',image)
