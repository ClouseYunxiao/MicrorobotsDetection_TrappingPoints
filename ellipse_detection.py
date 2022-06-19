import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
img=cv2.imread("test.jpg",3)
#img=cv2.blur(img,(1,1))
imgray=cv2.Canny(img,600,100,3)#
#cv2.imshow("0",imgray)
ret,thresh = cv2.threshold(imgray,127,255,cv2.THRESH_BINARY)
image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    if len(cnt)>50:
        S1=cv2.contourArea(cnt)
        ell=cv2.fitEllipse(cnt)
        S2 =math.pi*ell[1][0]*ell[1][1]
        if (S1/S2)>0.2 :
            img = cv2.ellipse(img, ell, (0, 255, 0), 2)
            print(str(S1) + "    " + str(S2)+"   "+str(ell[0][0])+"   "+str(ell[0][1]))
cv2.imshow("0",img)
cv2.waitKey(0)