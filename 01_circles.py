'''
https://www.pyimagesearch.com/2014/07/21/detecting-circles-images-using-opencv-hough-circles/
https://www.geeksforgeeks.org/find-circles-and-ellipses-in-an-image-using-opencv-python/

alternative:
https://www.geeksforgeeks.org/circle-detection-using-opencv-python/

SCANS:
600 dpi (-> Radius 100)
-5 Helligkeit

'''

import cv2
import numpy as np
import os


scale = 0.2
radius = 100
inpath = "images/scan/scan1.jpg"
outdir = "images/circles"

img = cv2.imread(inpath, cv2.IMREAD_COLOR)
width = img.shape[1]
height = img.shape[0]

# scale down, convert to grayscale and blur
scaled_width = int(img.shape[1] * scale)
scaled_height = int(img.shape[0] * scale)
scaled_img = cv2.resize(img, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA)
scaled_gray = cv2.cvtColor(scaled_img, cv2.COLOR_BGR2GRAY)
scaled_blurred = cv2.blur(scaled_gray, (5, 5))

# cv2.imshow("scaled_blurred", scaled_blurred)
# cv2.waitKey(0)

# Apply Hough transform on the blurred image.
detected_circles = cv2.HoughCircles(scaled_blurred,
                                    cv2.HOUGH_GRADIENT, 1, 20, param1=50,
                                    param2=30, minRadius=radius-1, maxRadius=radius+1)

print(len(detected_circles[0]), "circles found.")

if detected_circles is not None:
    # Convert the circle parameters x, y and r to integers.
    detected_circles = np.uint16(np.around(detected_circles))

    # for pt in detected_circles[0, :]:
    for i, pt in enumerate(detected_circles[0, :]):
        x, y, r = int(pt[0] / scale), int(pt[1] / scale), int(pt[2] / scale)

        # cv2.circle(img, (x, y), r, (0, 255, 0), 2)  # draw the circumference
        # cv2.circle(img, (x, y), 1, (0, 0, 255), 3)  # draw the centerpoint
        # cv2.imshow("Detected Circle", img)
        # cv2.waitKey(0)

        cropped_image = img[y-r:y+r, x-r:x+r]

        cropped_width = cropped_image.shape[1]
        cropped_height = cropped_image.shape[0]

        pad_right = r - width + x
        pad_left = r - x
        pad_top = r - y
        pad_bot = r - height + y

        pad_right = 0 if pad_right < 0 else pad_right
        pad_left = 0 if pad_left < 0 else pad_left
        pad_top = 0 if pad_top < 0 else pad_top
        pad_bot = 0 if pad_bot < 0 else pad_bot

        cropped_image = cv2.copyMakeBorder(cropped_image, pad_top, pad_bot, pad_left, pad_right,
                                           borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255))

        circle_stencil = np.zeros((2 * r, 2 * r, 3), dtype=np.uint8)
        cv2.circle(circle_stencil, (r, r), r-5, (255, 255, 255), -1, lineType=cv2.LINE_AA)

        masked_image = cv2.multiply(cropped_image / 255, circle_stencil / 255) * 255
        masked_image = np.array(masked_image, dtype='uint8')

        namesplit = os.path.splitext(os.path.basename(inpath))
        outpath = os.path.join(outdir, namesplit[0] + "_" + str(i) + ".png")
        cv2.imwrite(outpath, masked_image, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])

        # cv2.imshow("masked image", masked_image)
        # cv2.waitKey(0)
