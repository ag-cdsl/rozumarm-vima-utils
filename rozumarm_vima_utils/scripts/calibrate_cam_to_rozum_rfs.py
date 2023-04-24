import numpy as np
import scipy

import sys
from rozumarm_vima_utils.transform import rf_tf_c2r


# v0
# TABLE_FRAME_POINTS_FILEPATH = "assets/marker_points_in_rotated_table_frame_z_zero.npy"
# ROZUM_FRAME_POINTS_FILEPATH = "assets/marker_points_in_rozum_rf.npy"


# v1
TABLE_FRAME_POINTS_FILEPATH = "assets/aruco_corners.npy"
ROZUM_FRAME_POINTS_FILEPATH = "assets/marker_points_in_rozum_rf_v1.npy"


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
