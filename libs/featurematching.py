"""
BRISK Feature Matching
https://medium.com/analytics-vidhya/feature-matching-using-brisk-277c47539e8

HOMOGRAPHY
https://learnopencv.com/homography-examples-using-opencv-python-c/
https://www.programcreek.com/python/example/70443/cv2.WARP_INVERSE_MAP

PICKLE  alternative solution:
https://stackoverflow.com/questions/10045363/pickling-cv2-keypoint-causes-picklingerror
"""

import cv2
import numpy as np
import pickle

# set numpy to print w/o scientific notation
np.set_printoptions(suppress=True)


def get_detector(dect="sift", nfeatures=None, octaves=None, thresh=None):
    if dect == "orb":
        detector = cv2.ORB_create(nfeatures=nfeatures, edgeThreshold=thresh)
    elif dect == "akaze":
        detector = cv2.AKAZE_create(descriptor_type=None, descriptor_size=None, threshold=thresh)
    elif dect == "brisk":
        detector = cv2.BRISK_create(thresh=thresh, octaves=octaves)
        # norm = cv2.NORM_HAMMING
    else:
        if dect != "sift":
            print(f"Detector Type '{dect}' unknown, using SIFT instead.")
        detector = cv2.SIFT_create(nfeatures=nfeatures, nOctaveLayers=octaves, edgeThreshold=thresh)
    return detector


def detect(img, detector):
    # get keypoints and descriptors
    kp, desc = detector.detectAndCompute(img, None)
    return kp, desc


def match(kp1, desc1, kp2, desc2, detector=None, filter_ratio=0.75):
    # select flann algorithm based on feature type
    flann_algo_id = 1  # FLANN_INDEX_KDTREE -> cv2.SIFT
    if detector is not None:
        if (type(detector) == cv2.BRISK) or (type(detector) == cv2.ORB) or (type(detector) == cv2.AKAZE):
            flann_algo_id = 6  # FLANN_INDEX_LSH

    flann_params = dict(algorithm=flann_algo_id,
                        table_number=6,  # 12
                        key_size=12,  # 20
                        multi_probe_level=1)  # 2

    matcher = cv2.FlannBasedMatcher(flann_params, {})
    raw_matches = matcher.knnMatch(desc1, trainDescriptors=desc2, k=2)

    # # TODO: ALTERNATIVE: BruteForce Matcher
    # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # matches = bf.match(desB, desB)
    # matches = sorted(matches, key=lambda x: x.distance)
    # matched_image = cv2.drawMatches(imgA, kpA, imgB, kpB, matches, None, flags=2)

    # find good matches
    p1, p2, kp_pairs, good = filter_matches(kp1, kp2, raw_matches, ratio=filter_ratio)
    return p1, p2, kp_pairs, good


def filter_matches(kp1, kp2, matches, ratio=0.75):
    # Lows ratio test
    mkp1, mkp2, good= [], [], []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            m = m[0]
            mkp1.append( kp1[m.queryIdx] )
            mkp2.append( kp2[m.trainIdx] )
            good.append(m)

    p1 = np.float32([kp.pt for kp in mkp1])
    p2 = np.float32([kp.pt for kp in mkp2])
    kp_pairs = zip(mkp1, mkp2)
    return p1, p2, kp_pairs, good


def find_homography(p1, p2, ransac_threshold=5):
    # find homography
    if len(p1) >= 4:
        # enough points matched!
        H, status = cv2.findHomography(p1, p2, cv2.RANSAC, ransac_threshold)
        return H, status
    else:
        # not enough points found; exiting
        print('failed to find homography, not enough points.')
        return False


def scale_homography_matrix(matrix, template_scale=1., image_scale=1.):
    # SCALE TEMPLATE: divide column 0 and 1 by factor
    if template_scale != 1.:
        t_vector = np.array([template_scale, template_scale, 1])
        matrix = matrix / t_vector

    # SCALE IMAGE: multiply row 0 and 1 by factor
    if image_scale != 1.:
        i_vector = np.array([image_scale, image_scale, 1])
        matrix = matrix * i_vector[:, None]  # using broadcasting
        # alternatively using Transpose: (matrix.T * vector).T

    return matrix


def pickle_features(keypoints, descriptors, path):
    # make keypoints and descriptors "pickle*able"
    features = []
    for i, kp in enumerate(keypoints):
        feature = (kp.pt, kp.size, kp.angle, kp.response, kp.octave, kp.class_id, descriptors[i])
        features.append(feature)

    # save
    if path is not None:
        with open(path, 'wb') as file:
            pickle.dump(features, file)

    return features


def unpickle_features(path):
    with open(path, 'rb') as file:
        features = pickle.load(file)

    keypoints = []
    descriptors = []

    for feature in features:
        keypoint = cv2.KeyPoint(x=feature[0][0], y=feature[0][1], size=feature[1], angle=feature[2],
                                response=feature[3], octave=feature[4], class_id=feature[5])
        keypoints.append(keypoint)
        descriptors.append(feature[6])

    # convert list to numpy array
    descriptors = np.asarray(descriptors)
    return keypoints, descriptors
