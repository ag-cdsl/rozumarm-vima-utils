import time
import math

from scipy.spatial.transform import Rotation
from pulseapi import RobotPulse, position, MT_LINEAR

from .transform import rf_tf_v2r, map_tf_repr, map_gripper_rf


HOST = "http://10.10.10.20:8081"

OBJECT_HEIGHT = 0.03
Z_ZERO_LVL = 0.18 # for metal spatula: 0.2792491  # without handcrafted spatula: 0.124
Z_SWIPE_LVL = Z_ZERO_LVL + OBJECT_HEIGHT / 2
Z_PREP_LVL = Z_ZERO_LVL + 0.1
HOME_TCP_POS = (-0.3, 0, 0.6)  # (-0.3, 0, 0.3)
HOME_TCP_ANGLES = (math.pi, 0, 0)


class RozumArm:
    def __init__(self, use_mock_api=False):
        if use_mock_api:
            api_cls = MockAPI
        else:
            api_cls = RobotPulse
        
        self.api = api_cls(HOST)
        self.speed = 30.0

        self._move_home()
        self.api.open_gripper()

    def _wait(self):
        while self.api.status()["state"] != "ACTIVE":
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

    def swipe(self, posquat_1, posquat_2, from_rozumarm_rf: bool = False):
        """
        Swipe with end-effector from `posquat_1` to `posquat_2`,
        both specified in VIMA's reference frame.

        posquat_1: (2d pos, quat)
        posquat_2: (2d pos, quat)
        """
        if not from_rozumarm_rf:
            posquat_1, posquat_2 = [
                (rf_tf_v2r(pos), map_tf_repr(map_gripper_rf(quat)))
                for pos, quat in (posquat_1, posquat_2)
            ]

        pos_1, quat_1 = posquat_1
        pos_2, quat_2 = posquat_2
        pos_1, pos_2 = tuple(pos_1), tuple(pos_2)

        *_, z_rot_1 = Rotation.from_quat(quat_1).as_euler("XYZ")
        *_, z_rot_2 = Rotation.from_quat(quat_2).as_euler("XYZ")
        swipe_start_tcp_angles = (math.pi, 0, z_rot_1)
        swipe_stop_tcp_angles = (math.pi, 0, z_rot_2)

        self._move_tcp(pos=pos_1 + (Z_PREP_LVL,), angles=swipe_start_tcp_angles)
        self._move_tcp(pos=pos_1 + (Z_SWIPE_LVL,), angles=swipe_start_tcp_angles)
        self._move_tcp(pos=pos_2 + (Z_SWIPE_LVL,), angles=swipe_stop_tcp_angles)
        self._move_tcp(pos=pos_2 + (Z_PREP_LVL,), angles=swipe_stop_tcp_angles)
        self._move_home()


class MockAPI:
    def __init__(self, host):
        pass
    
    def status(self):
        return {'state': 'ACTIVE'}
    
    def get_position(self):
        return position([-0.3, 0., 0.2], [math.pi, 0., 0.])
    
    def set_position(self, position, *args, **kwargs):
        print(f'went through point {position.point} with rot {position.rotation}')
    
    def close_gripper(self):
        pass
