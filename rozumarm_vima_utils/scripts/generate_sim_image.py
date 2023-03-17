from typing import Callable
import sys
import pathlib

import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

sys.path.append(
    pathlib.Path(__file__).parents[2].as_posix()
)

from rozumarm_vima_utils.scene_renderer import VIMASceneRenderer
from rozumarm_vima_utils.robot import RozumArm
from rozumarm_vima_utils.scripts.detect_cubes import mock_detect_cubes
from rozumarm_vima_utils.transform import rf_tf_c2r, map_tf_repr_c2r


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
    tcp_pos = [getattr(pos.point, axis) for axis in ('x', 'y', 'z')]
    tcp_angles = [getattr(pos.rotation, angle_name) for angle_name in ('roll', 'pitch', 'yaw')]
    tcp_quat = Rotation.from_euler('xyz', tcp_angles).as_quat() 
    r.set_arm_state(tcp_pos, tcp_quat)
    
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
