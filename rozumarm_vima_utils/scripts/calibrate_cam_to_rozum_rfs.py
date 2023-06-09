import numpy as np
import scipy

import sys
from rozumarm_vima_utils.transform import rf_tf_c2r


# v2
# TABLE_FRAME_POINTS_FILEPATH = "assets/aruco_corners_top.npy"
# ROZUM_FRAME_POINTS_FILEPATH = "assets/marker_points_in_rozum_rf_v1.npy"

# v3 (synth alignment by 4.5 mm)
TABLE_FRAME_POINTS_FILEPATH = "assets/aruco_corners_top.npy"
ROZUM_FRAME_POINTS_FILEPATH = "assets/marker_points_in_rozum_rf_v2_synth.npy"


# v4 (temp test)
# import os
# TABLE_FRAME_POINTS_FILEPATH = os.path.expanduser("~/test.npy")
# ROZUM_FRAME_POINTS_FILEPATH = "assets/marker_points_in_rozum_rf_v2_synth.npy"


def find_rot():
    key_points_crf = np.load(TABLE_FRAME_POINTS_FILEPATH)
    key_points_crf = key_points_crf[:, :2]
    key_points_rrf = np.load(ROZUM_FRAME_POINTS_FILEPATH)

    points_rrf_3d = np.hstack((key_points_rrf, np.zeros((4, 1))))
    points_crf_3d = np.hstack((key_points_crf, np.zeros((4, 1))))

    # from rozumarm_vima_utils.transform import C2R_TF_SCALE
    # points_crf_3d *= C2R_TF_SCALE

    poins_rrf_centered = points_rrf_3d - points_rrf_3d.mean(axis=0)
    points_crf_centered = points_crf_3d - points_crf_3d.mean(axis=0)

    # a and b must be centered
    a = points_crf_centered  # for cam
    b = poins_rrf_centered  # for robot
    rot, rssd = scipy.spatial.transform.Rotation.align_vectors(a, b)
    return rot, rssd
    

def get_new_rot():
    rot, rssd = find_rot()
    angles = rot.as_euler('XYZ', degrees=True)
    print(f'New rot XYZ angles: {angles}')
    print(f'rssd: {rssd}')


def main():
    """
    use determined bias value only after setting scale
    """
    key_points_crf = np.load(TABLE_FRAME_POINTS_FILEPATH)
    key_points_crf = key_points_crf[:, :2]
    key_points_rrf = np.load(ROZUM_FRAME_POINTS_FILEPATH)

    # determine scale
    pdist_crf = scipy.spatial.distance.pdist(key_points_crf)
    pdist_rrf = scipy.spatial.distance.pdist(key_points_rrf)
    scale = (pdist_rrf / pdist_crf).mean()
    print(f'scale: {scale}')
    
    # determine CAM -> ROZUM bias
    key_points_crf_rot_to_r = np.vstack([
    rf_tf_c2r(key_points_crf[i], apply_bias=False)
        for i in range(4)
    ])

    rf_offsets = key_points_rrf - key_points_crf_rot_to_r
    avg_offset = rf_offsets.mean(axis=0)
    offset_std = rf_offsets.std(axis=0)
    print(f'\nrf offsets:\n{rf_offsets}')
    print(f'\nmean: {avg_offset}\nstd: {offset_std}')


if __name__ == '__main__':
    main()
    # get_new_rot()
