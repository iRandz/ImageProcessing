import numpy as np
import cv2

# !!!!!!!!!!!!! Needs new inputs before it works!
input_image = cv2.imread("traffic.png")
template = cv2.imread("biker.png")
output_image = input_image.copy()

############################################################
# Compute the Histogram Backprojection for the template on
# the traffic image. Use it to draw a bounding box around
# the biker in the traffic image.
# Hint: Use the built in calcBackProject.

# Your code here
input_image_hsv = cv2.cvtColor(input_image,cv2.COLOR_BGR2HSV)
template_hsv = cv2.cvtColor(template,cv2.COLOR_BGR2HSV)

ranges = [0, 180, 0, 256]
mask = cv2.inRange(template_hsv, np.array((0.0, 50.0, 30.0)), np.array((180.0, 255.0, 255.0)))

hist = cv2.calcHist([template_hsv], [0,1], mask, [180, 256], ranges)
cv2.normalize(hist, hist, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)


backproj = cv2.calcBackProject([input_image_hsv], [0, 1], hist, ranges, 1)
cv2.normalize(backproj, backproj, 0, 255, cv2.NORM_MINMAX)

cv2.imwrite("back_projection.png", backproj)

coord = np.where(backproj == backproj.max())
coord = list(zip(*coord))

template_size = template.shape

upper_left_corner = (coord[0][1] - template_size[1]//2, coord[0][0] - template_size[0]//2)
lower_right_corner = (coord[0][1] + template_size[1]//2, coord[0][0] + template_size[0]//2)
output_image = cv2.rectangle(output_image, upper_left_corner, lower_right_corner, (0,0,255))
############################################################

cv2.imwrite("output.png", output_image)