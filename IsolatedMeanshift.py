import numpy as np
import cv2
import OurMeanshift

image = cv2.imread("data/radial_gradient.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
img1 = np.copy(image)

track_window = [150, 150, 40, 40]
our_termination_crit = 1

x, y, w, h = track_window
img1 = cv2.rectangle(img1, (x, y), (x + w, y + h), 255, 2)
cv2.imshow('img1', img1)

track_window, yC, xC = OurMeanshift.our_meanShift(image, track_window, our_termination_crit)

print(track_window)
# Draw it on image
x, y, w, h = track_window
img2 = cv2.rectangle(image, (x, y), (x + w, y + h), 255, 2)

cv2.imshow('img2', img2)
cv2.waitKey(0)