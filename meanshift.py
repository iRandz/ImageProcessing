import numpy as np
import cv2
import OurMeanshift

cap = cv2.VideoCapture("data/slow_traffic_small.mp4")
template = cv2.imread("data/biker.png")

template_hsv = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)

ranges = [0, 180, 0, 256]
mask = cv2.inRange(template_hsv, np.array((0.0, 50.0, 30.0)), np.array((180.0, 255.0, 255.0)))

hist = cv2.calcHist([template_hsv], [0, 1], mask, [180, 256], ranges)
cv2.normalize(hist, hist, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

cv2.imshow('hist', hist)

# Video scrubbing parameters
frame_counter = 0
start_frame = 114
track_window = [592, 180, template.shape[1], template.shape[0]]
# track_window = [592, 180, 10, 10]

# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
termination_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.7)
our_termination_crit = 1
innerRet = True
while True:
    ret, frame = cap.read()
    if ret and innerRet:
        frame_counter += 1

        if frame_counter < start_frame:
            continue

        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        backproj = cv2.calcBackProject([frame_hsv], [0, 1], hist, ranges, 1)
        cv2.normalize(backproj, backproj, 0, 255, cv2.NORM_MINMAX)

        cv2.imshow('BackProjection', backproj)

        #backproj = cv2.GaussianBlur(backproj, (template.shape[1] - 1, template.shape[0] - 1), 0)
        backproj = cv2.GaussianBlur(backproj, (11, 11), 0)

        cv2.imshow('BlurredBackProjection', backproj)

        # innerRet, track_window = cv2.meanShift(backproj, track_window, termination_crit)
        track_window, yC, xC = OurMeanshift.our_meanShift(backproj, track_window, our_termination_crit)



        # Draw it on image
        yC = round(yC)
        xC = round(xC)
        x, y, w, h = track_window
        img2 = cv2.rectangle(frame, (x, y), (x + w, y + h), 255, 2)  # Box
        img2 = cv2.rectangle(img2, (xC, yC), (xC + 1, yC + 1), 255, 2)  # Center of mass
        cv2.imshow('img2', img2)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    else:
        break
