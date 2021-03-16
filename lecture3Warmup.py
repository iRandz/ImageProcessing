import cv2
import numpy as np

input_image = cv2.imread("kasper.png")
grey = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
output_image = grey.copy()
corners = input_image.copy()
############################################################
# Use cv2.cornerHarris to find corners and draw them with circles on output picture

# Your code here!
corners = cv2.cornerHarris(grey,2,3, 0.04)
coordinates = np.where(corners > 0.4 * corners.max())

for c in list(zip(*coordinates)):
  output_image = cv2.circle(input_image, (c[1], c[0]), 10, (0,0,255))

############################################################

cv2.imshow('output', output_image)
cv2.waitKey(0)