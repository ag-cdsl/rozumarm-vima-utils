import time
import math
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation
from pulseapi import RobotPulse, position, MT_LINEAR, pose

from .transform import rf_tf_v2r, map_tf_repr, map_gripper_rf


HOST = "http://10.10.10.20:8081"

OBJECT_HEIGHT = 0.03
Z_ZERO_LVL = 0.18 # for metal spatula: 0.2792491  # without handcrafted spatula: 0.124
Z_SWIPE_LVL = Z_ZERO_LVL + OBJECT_HEIGHT / 2
Z_PREP_LVL = Z_ZERO_LVL + 0.1
# HOME_TCP_POS = (-0.3, 0, 0.6)  # in front of base
HOME_TCP_POS = (0., 0.3, 0.6)  # top-camera-safe pos to the side
HOME_CAMERA_SAFE_POSE = (90., -90., 0, -180, -270, 0.)  # top-camera-safe
HOME_POSE = (180., -90., -30., -150, -270, 0.)
HOME_TCP_ANGLES = (math.pi, 0, 0)


class RozumArm:
    def __init__(self, use_mock_api=False):
        if use_mock_api:
            api_cls = MockAPI
        else:
            api_cls = RobotPulse
        
        self.api = api_cls(HOST)
        self.speed = 20.0

        self._move_home()
        # self.api.open_gripper()

    def _wait(self):
        
        while self.api.status().state != "ACTIVE":  # for pulse-api 1.6.0
        # self.api.status()["state"] != "ACTIVE":  # for pulse-api 1.8.4
            time.sleep(0.1)

    def _move_tcp(self, pos, angles):
        self.api.set_position(
            position(pos, angles),
            self.speed,
            motion_type=MT_LINEAR
        )
        self._wait()

    def _move_home(self):
        self._move_tcp(HOME_TCP_POS, HOME_TCP_ANGLES)
        # self.api.set_pose(pose(HOME_CAMERA_SAFE_POSE), speed=self.speed)
        # self._wait()

    def swipe(self, posquat_1, posquat_2, from_rozumarm_rf: bool = False):
        """
        Swipe with end-effector from `posquat_1` to `posquat_2`,
        both specified in VIMA's reference frame.

        posquat_1: (2d pos, quat)
        posquat_2: (2d pos, quat)
        """

        pos_1, quat_1 = posquat_1
        pos_2, quat_2 = posquat_2

        pos_1 = np.asarray(pos_1)
        pos_2 = np.asarray(pos_2)

        vec = np.float32(pos_2) - np.float32(pos_1)
        length = np.linalg.norm(vec)
        vec = vec / length
        
        # standard offsets
        # pos_1 -= vec * 0.02
        # pos_2 -= vec * 0.05
        
        pos_1 -= vec * 0.05
        pos_2 -= vec * 0.02

        posquat_1 = (pos_1, quat_1)
        posquat_2 = (pos_2, quat_2)

        if not from_rozumarm_rf:
            posquat_1, posquat_2 = [
                (rf_tf_v2r(pos), map_gripper_rf(map_tf_repr(quat)))
                for pos, quat in (posquat_1, posquat_2)
            ]

        pos_1, quat_1 = posquat_1
        pos_2, quat_2 = posquat_2
        pos_1, pos_2 = tuple(pos_1), tuple(pos_2)

        *_, z_rot_1 = Rotation.from_quat(quat_1).as_euler("XYZ")
        *_, z_rot_2 = Rotation.from_quat(quat_2).as_euler("XYZ")
        swipe_start_tcp_angles = (math.pi, 0, z_rot_1)
        swipe_stop_tcp_angles = (math.pi, 0, z_rot_2)

        # print('moving to PREP_HOME_POSE before swipe...')
        # self.api.set_pose(pose(HOME_POSE), speed=self.speed)
        # self._wait()

        # print(f'starting swipe at {pos_1}')  # hw failure here
        self._move_tcp(pos=pos_1 + (Z_PREP_LVL,), angles=swipe_start_tcp_angles)
        self._move_tcp(pos=pos_1 + (Z_SWIPE_LVL,), angles=swipe_start_tcp_angles)
        self._move_tcp(pos=pos_2 + (Z_SWIPE_LVL,), angles=swipe_stop_tcp_angles)
        self._move_tcp(pos=pos_2 + (Z_PREP_LVL,), angles=swipe_stop_tcp_angles)

        # print('moving to PREP_HOME_POSE after swipe...')
        # self.api.set_pose(pose(HOME_POSE), speed=self.speed)  # hw failure here
        # self._wait()

        self._move_home()

    @staticmethod
    def get_swipe_quat(pos_1, pos_2):
        """
        positions are in VIMA-rf
        quat is in VIMA-rf
        """
        pos_1 = np.asarray(pos_1)
        pos_2 = np.asarray(pos_2)
        
        dir_vec = pos_2 - pos_1
        # theta = np.arctan2(dir_vec[1], dir_vec[0])
        theta = np.arctan(dir_vec[1] / dir_vec[0])  # for shortest rotation
        quat = Rotation.from_euler('XYZ', (0, 0, theta)).as_quat()
        return quat


class MockAPI:
    def __init__(self, host):
        pass
    
    def status(self):
        return SimpleNamespace(state="ACTIVE")
    
    def get_position(self):
        return position([-0.3, 0., 0.2], [math.pi, 0., -math.pi / 4])
    
    def _get_triplet_repr(self, triplet):
        return '({})'.format(', '.join([f'{x:.3f}' for x in triplet]))
        
    def set_position(self, position, *args, **kwargs):
        coords = [getattr(position.point, attr_name) for attr_name in ('x', 'y', 'z')]
        angles = [getattr(position.rotation, attr_name) for attr_name in ('pitch', 'roll', 'yaw')]
        
        coords_repr = self._get_triplet_repr(coords)
        angles_repr = self._get_triplet_repr(angles)
        print(f'went through point xyz: {coords_repr} with rot pry: {angles_repr}')
    
    def open_gripper(self):
        pass
    
    def close_gripper(self):
        pass
