import glob
import cv2
import numpy as np
# Background subtraction or something like that
# Need to add images!!!!!!!!! None actually available right now!
background_images = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in glob.glob("background_images/*.tif")]

test_image = cv2.imread("test2.tif", cv2.IMREAD_GRAYSCALE)

background_images = np.asarray(background_images)
background = np.median(background_images, axis=0)

foreground = np.uint8(np.abs(background-test_image))

background = np.uint8(background)
# Thresh to 200
_,output_image = cv2.threshold(foreground, 40, 255, cv2.THRESH_BINARY)
red_image = cv2.cvtColor(test_image, cv2.COLOR_GRAY2RGB)

# use output_image as the red channel for red_image but only where values are 255.
red_image[:,:][:,:,2] = np.where(output_image == 255, output_image, red_image[:,:][:,:,2])
cv2.imshow("red", red_image)
cv2.imshow("foreground", foreground)
cv2.imshow("background", background)
cv2.imshow("test", test_image)
cv2.imshow("output", output_image)
cv2.waitKey()