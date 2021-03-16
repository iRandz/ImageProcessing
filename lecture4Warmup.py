import glob
import cv2
import numpy as np

# No images currently available!!!!!! Would need to add new ones.
background_images = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in glob.glob("background_images/*.tif")]

test_image = cv2.imread("test_image.png", cv2.IMREAD_GRAYSCALE)

############################################################
# Return a mask for the people in test_image.png (see
# desired_output.png for an example)
# Compute the median of background_images and use the resulting
# image as a base for doing background subtraction on test_image.

# Your code here
thresh = 50

median_image = np.median(background_images, 0)
cv2.imwrite("median.png", median_image)
diff = np.abs(test_image - median_image)

# output_image = np.where(diff > thresh, 255)

mean_image = np.mean(background_images, 0)
cv2.imwrite("mean.png", mean_image)
diff2 = np.abs(test_image - mean_image)

_, output_image = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)
_, output_image2 = cv2.threshold(diff2, thresh, 255, cv2.THRESH_BINARY)

############################################################

cv2.imwrite("output.png", output_image)
cv2.imwrite("output2.png", output_image2)