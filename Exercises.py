import cv2
import numpy as np

# from matplotlib import pyplot as plt

img = cv2.imread('neon-text.png')
print(img.dtype)
imgGrey = np.zeros([img.shape[0], img.shape[1], 1], np.uint8)

print(img[100, 100, 0])
print(img[100, 100, 1])
print(img[100, 100, 2])

#print((img[100, 100, 0] + img[100, 100, 1] + img[100, 100, 2]))

# height, width, channel = img.shape
# for i in range(height):
#     for j in range(width):
#         R = float(img[i, j, 2])
#         G = float(img[i, j, 2])
#         B = float(img[i, j, 2])
#         imgGrey[i, j] = np.uint8((R + B + G) / 3)

#grey = 0.07 * img[:,:,2] + 0.72 * img[:,:,1] + 0.21 * img[:,:,0]
#imgGrey = grey.astype(np.uint8)

#grey = img.dot([0.07, 0.72, 0.21])
#imgGrey = grey.astype(np.uint8)

imgGrey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

print(imgGrey[100, 100])

cv2.imshow('img', imgGrey)
cv2.waitKey(0)

# for i in range(img.shape[0]):
#   for j in range(img.shape[1]):
#      print(img[i, j])
