"""
https://learnopencv.com/blob-detection-using-opencv-python-c/
https://pythonexamples.org/python-opencv-cv2-find-contours-in-image/
https://www.pyimagesearch.com/2016/10/31/detecting-multiple-bright-spots-in-an-image-with-python-and-opencv/
https://docs.opencv.org/3.4.15/da/d0c/tutorial_bounding_rects_circles.html
"""


import cv2
import numpy as np
import os


def get_imagepaths(path, recursive=False):
    imagepaths = []
    for root, dirs, files in os.walk(path, topdown=True):
        for file in files:
            imagepaths.append(os.path.join(root, file))
        if not recursive:
            break
    return imagepaths


def get_bbox_and_circle(contours):
    # Approximate contours to polygons + get bounding rects and circles
    contours_poly = [None] * len(contours)
    boundRect = [None] * len(contours)
    centers = [None] * len(contours)
    radius = [None] * len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])
        centers[i], radius[i] = cv2.minEnclosingCircle(contours_poly[i])
    return contours_poly, boundRect, centers, radius


def draw_bbox_and_circle(img, contours_poly, boundRect, centers, radius):
    image = img.copy()
    for i in range(len(contours_poly)):
        color = (0, 255, 0)  # (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))
        cv2.drawContours(image, contours_poly, i, color)
        cv2.rectangle(image, (int(boundRect[i][0]), int(boundRect[i][1])),
                      (int(boundRect[i][0] + boundRect[i][2]), int(boundRect[i][1] + boundRect[i][3])), color, 2)
        cv2.circle(image, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), color, 2)
    return image


dilate_px = 3
erode_px = 2
blur = 9
thresh = 235
minArea = 5000

indir = "images/circles"
outdir = "images/symbols/_raw"
imagepaths = get_imagepaths(indir)

for imagepath in imagepaths:
    image = cv2.imread(imagepath, -1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur, blur), 0)

    eroded = cv2.erode(blurred, None, iterations=dilate_px)
    dilated = cv2.dilate(eroded, None, iterations=erode_px)

    thresh_img = cv2.threshold(dilated, thresh, 255, cv2.THRESH_BINARY)[1]

    mode = cv2.RETR_TREE
    method = cv2.CHAIN_APPROX_SIMPLE
    contours, hierarchy = cv2.findContours(thresh_img.copy(), mode, method)


    inner_contours = []
    for i, contour_hierarchy in enumerate(hierarchy[0]):
        if contour_hierarchy[3] != 0:  # hierarchy row 3 is parent-ID
            inner_contours.append(i)
        else:
            M = cv2.moments(contours[i])
            area = M['m00']

            # # calculate centerpoint
            # cX = int(M["m10"] / area)
            # cY = int(M["m01"] / area)

            if area < minArea:
                inner_contours.append(i)


    # reverse order so that last indices get removed first
    inner_contours.reverse()
    for i in inner_contours:
        contours.pop(i)

    contours_poly, boundRect, centers, radius = get_bbox_and_circle(contours)

    # # DRAW PREVIEW
    # preview = image.copy()
    # # contours
    # cv2.drawContours(preview, contours, -1, (255, 0, 0), 2)
    # # bounding boxes and circles
    # preview = draw_bbox_and_circle(preview, contours_poly, boundRect, centers, radius)
    # cv2.imshow("preview", preview)
    # cv2.waitKey(0)

    for i, contour in enumerate(contours):  # #for i in range(len(contours)):
        r = int(radius[i])
        x = int(centers[i][0])
        y = int(centers[i][1])

        # create mask from contour and crop it to enclosing circle
        mask = np.ones((image.shape[1], image.shape[0], 1), np.uint8)*255
        cv2.drawContours(mask, [contour], -1, (0, 0, 0), -1)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

        masked_img = cv2.addWeighted(image, 1, mask, 1, 0.0)

        y1 = y - r
        y2 = y + r
        x1 = x - r
        x2 = x + r

        px1 = 0 if x1 >= 0 else -x1
        py1 = 0 if y1 >= 0 else -y1
        px2 = 0 if x2 < image.shape[1] else x2 - image.shape[1]
        py2 = 0 if y2 < image.shape[0] else y2 - image.shape[0]
        # print(px1, px2, py1, py2)

        # extend canvas if circle is too big
        if px1 + px2 + py1 + py2 > 0:
            masked_img = cv2.copyMakeBorder(masked_img, py1, py2, px1, px2,
                                            borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255))
            py2 += py1
            px2 += px1

        cropped_img = masked_img[y1+py1: y2+py2, x1+px1: x2+px2]

        # cv2.imshow("mask", masked_img)
        # cv2.waitKey(0)

        namesplit = os.path.splitext(os.path.basename(imagepath))
        outpath = os.path.join(outdir, namesplit[0] + "_" + str(i) + ".png")

        cv2.imwrite(outpath, cropped_img, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])

        # cv2.imshow("cropped_img", cropped_img)
        # cv2.waitKey(0)
