from typing import Callable

import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation


from rozumarm_vima_utils.scene_renderer import VIMASceneRenderer
from rozumarm_vima_utils.robot import RozumArm
from rozumarm_vima_utils.scripts.detect_cubes import mock_detect_cubes
from rozumarm_vima_utils.transform import (
    rf_tf_c2r,
    map_tf_repr_c2r,
    rf_tf_r2v,
    map_tf_repr,
    map_gripper_rf
)


def render_real2sim(r: VIMASceneRenderer, robot: RozumArm, cubes_detector: Callable):
    """
    r: scene renderer
    """
    obj_posquats = cubes_detector()

    # map from cam to rozum
    obj_posquats = [
        (rf_tf_c2r(pos), map_tf_repr_c2r(quat)) for pos, quat in obj_posquats
    ]

    # set objs pos
    _ = r.render_scene(obj_posquats)
    
    # set arm pos
    pos = robot.api.get_position()
    rozum_tcp_pos = [getattr(pos.point, axis) for axis in ('x', 'y', 'z')]
    rozum_tcp_angles = [getattr(pos.rotation, angle_name) for angle_name in ('roll', 'pitch', 'yaw')]
    rozum_tcp_quat = Rotation.from_euler('XYZ', rozum_tcp_angles).as_quat()
    
    # map from rozum to vima
    vima_tcp_pos = rf_tf_r2v(rozum_tcp_pos, from3d=True)
    tmp_tcp_quat = map_tf_repr(rozum_tcp_quat)
    vima_tcp_quat = map_gripper_rf(tmp_tcp_quat)
    
    r.set_arm_state(vima_tcp_pos, vima_tcp_quat)
    
    image = r.render_full_view()
    return image


def main():
    r = VIMASceneRenderer('sweep_without_exceeding', hide_arm_rgb=False)
    robot = RozumArm(use_mock_api=True)

    r.reset(exact_num_swept_objects=1)
    image = render_real2sim(r, robot, mock_detect_cubes)
    plt.imsave('real2sim.png', image)


if __name__ == '__main__':
    main()
