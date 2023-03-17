from typing import Callable
import sys
import pathlib

import numpy as np

sys.path.append(
    pathlib.Path(__file__).parents[2].as_posix()
)

from rozumarm_vima_utils.transform import rf_tf_c2r, map_tf_repr_c2r


def run_loop(r, robot, oracle, cubes_detector: Callable, n_iters=3):
    """
    r: scene renderer
    """
    for _ in range(n_iters):
        obj_posquats = cubes_detector()

        # map from cam to rozum
        obj_posquats = [
            (rf_tf_c2r(pos), map_tf_repr_c2r(quat)) for pos, quat in obj_posquats
        ]

        front_img, top_img = r.render_scene(obj_posquats)
        obs = {
            'rgb': {
                'front': np.transpose(front_img, axes=(2, 0, 1)),
                'top': np.transpose(top_img, axes=(2, 0, 1))
            },
            'ee': 1  # spatula
        }

        action = oracle.act(obs)
        if action is None:
            print("ORACLE FAILED.")
            return

        clipped_action = {
            k: np.clip(v, r.env.action_space[k].low, r.env.action_space[k].high)
            for k, v in action.items()
        }

        posquat_1 = (clipped_action["pose0_position"], clipped_action["pose0_rotation"])
        posquat_2 = (clipped_action["pose1_position"], clipped_action["pose1_rotation"])
        robot.swipe(posquat_1, posquat_2)


def main():
    from rozumarm_vima_utils.scene_renderer import VIMASceneRenderer
    from rozumarm_vima_utils.robot import RozumArm
    from rozumarm_vima_utils.scripts.detect_cubes import mock_detect_cubes

    r = VIMASceneRenderer('sweep_without_exceeding')
    oracle = r.env.task.oracle(r.env)
    robot = RozumArm(use_mock_api=True)
    
    r.reset(exact_num_swept_objects=1)
    
    run_loop(r, robot, oracle, cubes_detector=mock_detect_cubes, n_iters=5)


if __name__ == '__main__':
    main()
