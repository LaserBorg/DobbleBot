"""
based on my OEKOTEX featurematching project
"""

import os
from libs.featurematching import *
from libs.image import transform_img, draw_matches, draw_features, get_dst
from libs.odometry import get_vectors2


def main(photo_path, reference_directory_path):
    # load photo
    photo_orig = cv2.imread(photo_path)
    photo_name = os.path.splitext(os.path.basename(photo_path))[0]

    # scale photo down to max-resolution
    photo_scaled, vector = transform_img(photo_orig, scale_long=max_resolution, allow_upscale=False, return_vector=True)
    imagescale = vector[2]

    # load all templates in directory
    root = ""
    template_list = []
    for root, _, templates in os.walk(reference_directory_path, topdown=True):
        for template in templates:
            if os.path.splitext(template)[1] == ".png":
                template_list.append(template)
        break

    print(f"found {len(template_list)} templates : {template_list} \n")

    # # init history [(kp1, kp2, good, H, status), (...)] and sum-list [sum(status), ...]
    # history = []
    # sums = []

    print("template :\tinliers / matched")
    print("-----------------------------")

    # get detector and extract features from photo
    image_detector = get_detector(dect, nfeatures, octaves, detector_thresh)
    symbol_detector = get_detector(dect, nfeatures_symbol, octaves, detector_thresh)
    kp2, desc2 = detect(photo_scaled, image_detector)

    # LOOP OVER ALL SYMBOLS
    for template_name in template_list:
        template_path = os.path.join(root, template_name)
        img_template = cv2.imread(template_path)

        # SAVE KEYPOINTS AS PICKLE
        pickle_path = os.path.join(symbols_dir, "pickle", os.path.splitext(template_name)[0] + "-" + dect + ".pkl")

        if load_from_pickle:
            kp1, desc1 = unpickle_features(pickle_path)
        else:
            # extract features from template and pickle its keypoints and descriptors
            kp1, desc1 = detect(img_template, symbol_detector)
            pickle_features(kp1, desc1, pickle_path)

        # match features
        p1, p2, kp_pairs, good = match(kp1, desc1, kp2, desc2, image_detector, filter_ratio)

        # find Homography Matrix
        H_scaled, status = find_homography(p1, p2, ransac_threshold=reproject_thresh)

        sum_status = np.sum(status)
        # history.append((kp1, kp2, good, H_scaled, status))
        # sums.append(sum_status)

        print(f"{os.path.splitext(template_name)[0]} :\t{sum_status} / {len(status)}")

        # skip symbol if not enough matched features or RANSAC inliers
        if len(status) < 10 or sum_status < 5:
            continue

        # draw rich features
        draw_features(img_template, kp1, show="",  # "template_contrasted",
                      save=save_feature_imgs, save_path="images/results/templateFeatures.jpg")
        draw_features(photo_scaled, kp2, show="",  # "photo_contrasted",
                      save=save_feature_imgs, save_path="images/results/imageFeatures.jpg")

        # draw matches and get cornerpoints of symbol  TODO: hacky; don't do that it the same function :D
        match_img, cornerpoints = draw_matches(img_template, photo_scaled, kp1, kp2, good, status, H_scaled, draw_lines)

        cv2.imshow("matches", match_img)
        matches_path = os.path.join(output_dir, photo_name + " matches.jpg")
        cv2.imwrite(matches_path, match_img)

        # calculate scaled homography matrix and image size for bigger inputs and outputs
        H = scale_homography_matrix(H_scaled, template_scale=templatescale, image_scale=1/imagescale)

        template_height, template_width, _ = img_template.shape  # height, width, channels
        scaled_template_size = (int(template_width*templatescale), int(template_height*templatescale))

        # unwarp fullsize photo towards upscaled template dimensions
        scaled_unwarped_img = cv2.warpPerspective(photo_orig, H, scaled_template_size,
                                                  flags=cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC)

        # # scale copy of unwarped image back to template size
        # unwarped_img = transform_img(scaled_unwarped_img, scale=1/templatescale, interpolation=cv2.INTER_CUBIC)

        cv2.imshow("", scaled_unwarped_img)

        rvecs, tvecs, cmat = get_vectors2(photo_scaled, cornerpoints)
        print("rvecs:", rvecs)
        print("tvecs:", tvecs)

        cv2.waitKey(0)


    # # select H and status of template with highest feature count
    # max_index = sums.index(max(sums))
    # kp1, kp2, good, H, status = history[max_index]
    #
    # # reload selected template image
    # selected_image = template_list[max_index]
    # template_path = os.path.join(root, selected_image)
    # img_template = cv2.imread(template_path)
    #
    # print("selecting template:", os.path.splitext(selected_image)[0])


# -----------------------------------------------------------


imagepath = "images/dobble_01_1000.jpg"
max_resolution = 1200

symbols_dir = "images/symbols/"
load_from_pickle = False

output_dir = "images/results/"      # debug images

# detection
dect = "sift"  # detector Type: "sift" "brisk" "akaze" "orb"
nfeatures = 8000
nfeatures_symbol = 1000
detector_thresh = 15
octaves = 10

# matching
filter_ratio = 0.75

# homography: RANSAC reprojection threshold
reproject_thresh = 5

# supersample image
templatescale = 1.

# visualisation / output
draw_lines = True
save_feature_imgs = False


if __name__ == "__main__":
    main(imagepath, symbols_dir)
