"""
https://rdmilligan.wordpress.com/2015/10/15/augmented-reality-using-opencv-opengl-and-blender/
https://learnopencv.com/head-pose-estimation-using-opencv-and-dlib/
"""

import numpy as np
import cv2


# def order_points(points):
#     s = points.sum(axis=1)
#     diff = np.diff(points, axis=1)
#
#     ordered_points = np.zeros((4, 2), dtype="float32")
#
#     ordered_points[0] = points[np.argmin(s)]
#     ordered_points[2] = points[np.argmax(s)]
#     ordered_points[1] = points[np.argmin(diff)]
#     ordered_points[3] = points[np.argmax(diff)]
#
#     return ordered_points
#
#
# def get_vectors(image, points, calibration_file='webcam_calibration_ouput.npz'):
#
#     # order points
#     points = order_points(points)
#
#     # load calibration data
#     with np.load(calibration_file) as X:
#         mtx, dist, _, _ = [X[i] for i in ('mtx' ,'dist' ,'rvecs' ,'tvecs')]
#
#     # set up criteria, image, points and axis
#     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
#
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#     imgp = np.array(points, dtype="float32")
#
#     objp = np.array([[0. ,0. ,0.] ,[1. ,0. ,0.],
#                      [1. ,1. ,0.] ,[0. ,1. ,0.]], dtype="float32")
#
#     # calculate rotation and translation vectors
#     cv2.cornerSubPix(gray ,imgp ,(11 ,11) ,(-1 ,-1) ,criteria)
#     rvecs, tvecs, _ = cv2.solvePnPRansac(objp, imgp, mtx, dist)
#
#     return rvecs, tvecs


def get_vectors2(image, image_points):

    image_points = image_points.astype('double')

    size = image.shape

    # 3D plane points
    model_points = np.array([[0., 0., 0.], [100., 0., 0.], [100., 100., 0.], [0., 100., 0.]], dtype="float32")

    # Camera internals
    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)

    camera_matrix = np.array([[focal_length, 0, center[0]],
                              [0, focal_length, center[1]],
                              [0, 0, 1]], dtype="double")

    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                  dist_coeffs, flags=cv2.SOLVEPNP_UPNP)

    # Project a 3D point (0, 0, 1000.0) onto the image plane.
    # We use this to draw a line sticking out of the nose

    nose_end_point2D, jacobian = cv2.projectPoints(np.array([(0.0, 0.0, 50.0)]),
                                                   rotation_vector, translation_vector, camera_matrix, dist_coeffs)

    for p in image_points:
        cv2.circle(image, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

    p1 = (int(image_points[0][0]), int(image_points[0][1]))
    p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

    cv2.line(image, p1, p2, (255, 0, 0), 2)

    # Display image
    cv2.imshow("Output", image)
    #cv2.waitKey(0)

    return rotation_vector, translation_vector, camera_matrix


if __name__ == "__main__":
    image = cv2.imread("../images/dobble_01.jpg")

    image_points = np.array([(359, 391),
                             (399, 561),
                             (337, 297),
                             (513, 301)], dtype="double")
    print(image_points)


    rvec, tvec, cmat = get_vectors2(image, image_points)

    print(f"Camera Matrix :\n {format(cmat)}")
    print(f"Rotation Vector:\n {format(rvec)}")
    print(f"Translation Vector:\n {format(tvec)}")
