"""
Useful files:

1) rozumarm_vima/scripts/run_model_loop.py
2) rozumarm-vima-utils/rozumarm_vima_utils/notebooks/itest_robot.ipynb
"""

import time

from scipy.spatial.transform import Rotation as R

from rozumarm_vima_utils.robot import RozumArm


POINTS = [
    # center
    (
        [-0.1955730723827631, -0.11038192678679404, 0.6906691404152611],
        [-3.0869227407263193, -1.240679527697654, -3.0425004166458525]
    ),
    # left
    (
        [-0.1955730723827631, -0.25, 0.65],
        [-3.1, -1.2, -3.0425004166458525]
    ),
    # right
    (
        [-0.4177575034681805, 0.2402043210249394, 0.4167568825958069],
        [2.5661427730657667, -0.8154069583456097, 2.3613085549435513]
    ),
]


def main():
    robot = RozumArm(use_mock_api=False)
    
    point_idx = 0
    while True:
        pos, angles_rad_to_api = POINTS[point_idx]

        robot.move_tcp(pos, angles_rad_to_api)
        time.sleep(1)

        point_idx = (point_idx + 1) % len(POINTS)


if __name__ == "__main__":
    main()
