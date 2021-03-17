import numpy as np
import cv2


def our_meanShift(inProj, window, criteria):
    # Get variables

    cur_rect = window
    outputCenterOfMassY = 0
    outputCenterOfMassX = 0

    # Check if legitimate window
    if window[3] <= 0 or window[2] <= 0:
        print("Input window has non-positive sizes")

    # establish window with values from backprojection
    # window = window & Rect(0, 0, size.width, size.height)

    # Establish max iterations and stopping criteria
    niters = 1000
    stop = criteria

    for i in range(0, niters):

        # If no window make new one
        if cur_rect is 0:
            cur_rect[1] = inProj.size.width / 2
            cur_rect[0] = inProj.size.height / 2

        cur_rect[1] = np.maximum(cur_rect[1], 1)
        cur_rect[0] = np.maximum(cur_rect[0], 1)

        # calculate center of mass
        xEnd = int(cur_rect[0] + cur_rect[2])
        yEnd = int(cur_rect[1] + cur_rect[3])
        xMass = 0
        yMass = 0
        totalMass = 0

        for y in range(int(cur_rect[1]), yEnd):
            xt = 0
            for x in range(int(cur_rect[0]), xEnd):
                yMass += inProj[y][x] * y
                xMass += inProj[y][x] * x
                totalMass += inProj[y][x]

        if totalMass < 1:
            print("No Matching pixels found")
            break
        centerOfMassY = yMass / totalMass
        centerOfMassX = xMass / totalMass

        nx = np.minimum(np.maximum(centerOfMassX, 0), inProj.shape[1] - cur_rect[2])
        ny = np.minimum(np.maximum(centerOfMassY, 0), inProj.shape[0] - cur_rect[3])

        outputCenterOfMassX = nx
        outputCenterOfMassY = ny

        nx = nx - cur_rect[2]/2
        ny = ny - cur_rect[3]/2

        dx = nx - cur_rect[0]
        dy = ny - cur_rect[1]

        cur_rect[0] = round(nx)
        cur_rect[1] = round(ny)

        # Did we move a significant amount?
        if dx * dx + dy * dy < stop:
            print("Iterations: ", i)
            break

    # Draw it on image
    x, y, w, h = cur_rect
    imgInner = cv2.rectangle(np.copy(inProj), (x, y), (x + w, y + h), 255, 2)
    cv2.imshow('imgInner', imgInner)

    window = cur_rect
    return window, outputCenterOfMassY, outputCenterOfMassX
