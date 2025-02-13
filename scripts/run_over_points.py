"""
Useful files:

1) rozumarm_vima/scripts/run_model_loop.py
2) rozumarm-vima-utils/rozumarm_vima_utils/notebooks/itest_robot.ipynb
"""


import numpy as np
from rozumarm_vima_utils.robot import RozumArm


POINTS = [
    ((0., 0.3, 0.6), (np.pi, 0, 0)),
    ((0., 0.3, 0.6), (np.pi, 0, 0)),
    ((0., 0.3, 0.6), (np.pi, 0, 0)),
]


def main():
    robot = RozumArm(use_mock_api=True)
    
    point_idx = 0
    while True:
        pos, angles = POINTS[point_idx]
        
        robot.move_tcp(pos, angles)
        
        point_idx = (point_idx + 1) % len(POINTS)


if __name__ == "__main__":
    main()
