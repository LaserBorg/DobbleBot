"""
CLAHE
https://www.geeksforgeeks.org/clahe-histogram-eqalization-opencv/
"""

import cv2
import numpy as np


# TODO: rotate, translate, adjust_canvas=True
def transform_img(img, translate=(0, 0), rotate=0, scale=1., scale_long=0, allow_upscale=True,
                  adjust_canvas=True, return_vector=False, interpolation=cv2.INTER_CUBIC):
    # init result image
    transformed_img = img.copy()

    # scale to max-resolution
    if scale_long > 0:
        img_height, img_width, img_channels = img.shape

        # check if either downscaling or allow_upscale
        if img_height > scale_long or img_width > scale_long or allow_upscale is True:
            if img_height > img_width:
                scale *= scale_long / img_height
            else:
                scale *= scale_long / img_width

    if scale != 1.:
        transformed_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=interpolation)

    if return_vector:
        vector = (translate, rotate, scale)
        return transformed_img, vector
    else:
        return transformed_img


def clahe(img, clipLimit=2., radius=8):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe_filter = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(radius, radius))
    cl = clahe_filter.apply(l)

    lab = cv2.merge((cl, a, b))
    result_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return result_img


def threshold(img, blocksize=199, constant=5):
    # use green channel
    gray = img[:, :, 1]

    # gray = cv2.convertScaleAbs(gray, alpha=1.8, beta=0)
    # gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    gray = cv2.adaptiveThreshold(gray, 255,
                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY,
                                 blocksize, constant)
    return gray


def draw_features(img, kp, show="", save=False, save_path="features.jpg", rich=True):
    # draw rich or basic feature image
    if rich:
        draw_keypoints_flag = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    else:
        draw_keypoints_flag = None

    result_img = cv2.drawKeypoints(img, kp, None, flags=draw_keypoints_flag)

    if type(show) is str and show != "":
        cv2.imshow(show, result_img)

    if save:
        cv2.imwrite(save_path, result_img)

    return result_img


def get_dst(img, H):
    h1, w1 = img.shape[:2]
    pts = np.float32([[0, 0], [0, h1 - 1], [w1 - 1, h1 - 1], [w1 - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, H)
    dst = np.int32(dst)
    return dst


def draw_matches(img1, img2, kp1, kp2, good, status=None, H=None, draw_lines=True):
    # draw homography
    dst = get_dst(img1, H)

    match_img = cv2.polylines(img2.copy(), [dst], True, 255, 2, cv2.LINE_AA)

    cornerpoints = np.array([(0, 0), (0, 0), (0, 0), (0, 0)], dtype="int32")

    for i in range(len(dst)):
        x = dst[i][0][0]
        y = dst[i][0][1]
        cornerpoints[i][0] = x
        cornerpoints[i][1] = y
        match_img = cv2.circle(match_img, (x, y), 3, (255, 0, 0), -1)

    # draw lines between matches
    if draw_lines is True:
        matches_mask = status.ravel().tolist()
        draw_params = dict(matchColor=(0,255,0), # draw matches in green color
                           singlePointColor=None,
                           matchesMask=matches_mask,  # draw only inliers
                           flags=2)
        match_img = cv2.drawMatches(img1, kp1, match_img, kp2, good, None, **draw_params)

    return match_img, cornerpoints
