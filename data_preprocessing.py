import numpy as np
import cv2

image = cv2.imread("H:\\img\\lena.jpg")
image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imshow("Original",image)
cv2.waitKey()


lap = cv2.Laplacian(image,cv2.CV_64F)
lap = np.uint8(np.absolute(lap))
cv2.imshow("Laplacian",lap)
cv2.waitKey()
