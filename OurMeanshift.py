import numpy as np
import cv2


def our_meanShift(inProj, window, criteria):
    # Get variables

    cur_rect = window

    print(cur_rect)

    # Check if legitimate window
    if window[3] <= 0 or window[2] <= 0:
        print("Input window has non-positive sizes")

    # establish window with values from backprojection
    # window = window & Rect(0, 0, size.width, size.height)

    # Establish max iterations and stopping criteria
    niters = criteria
    stop = 50

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
            for x in range(int(cur_rect[0]), xEnd):
                yMass += inProj[y][x] * y
                xMass += inProj[y][x] * x
                totalMass += inProj[y][x]

        if totalMass < 1:
            print("No Matching pixels found")
            break
        centerOfMassY = yMass / totalMass
        centerOfMassX = xMass / totalMass

        print(centerOfMassX)
        print(centerOfMassY)
        nx = np.minimum(np.maximum(centerOfMassX, 0), inProj.shape[1] - cur_rect[2])
        ny = np.minimum(np.maximum(centerOfMassY, 0), inProj.shape[0] - cur_rect[3])

        dx = nx - cur_rect[0]
        dy = ny - cur_rect[1]
        cur_rect[0] = int(nx)
        cur_rect[1] = int(ny)

        # Did we move a significant amount?
        if dx * dx + dy * dy < stop:
            break

    window = cur_rect
    return window
