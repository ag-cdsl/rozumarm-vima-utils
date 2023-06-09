from typing import Callable

import numpy as np

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

        pos_0 = clipped_action["pose0_position"]
        pos_1 = clipped_action["pose1_position"]
        eef_quat = robot.get_swipe_quat(pos_0, pos_1)
        
        # x_compensation_bias = 0.03
        # pos_0[0] += x_compensation_bias
        # pos_1[0] += x_compensation_bias
        
        posquat_0 = (pos_0, eef_quat)
        posquat_1 = (pos_1, eef_quat)
        robot.swipe(posquat_0, posquat_1)


class MockObjDetector:
    def __call__(self):
        obj_posquats = [
            ((0.3, -0.05), (0, 0, 0, 1)),
            ((0.0, -0.15), (0, 0, 0, 1))
        ]
        return obj_posquats


from rozumarm_vima_utils.scene_renderer import VIMASceneRenderer
def main():
    from rozumarm_vima_utils.robot import RozumArm

    r = VIMASceneRenderer('sweep_without_exceeding')
    oracle = r.env.task.oracle(r.env)
    robot = RozumArm(use_mock_api=True)
    
    r.reset(exact_num_swept_objects=1)
    
    detector = MockObjDetector()
    
    run_loop(r, robot, oracle, cubes_detector=detector, n_iters=5)


if __name__ == '__main__':
    main()
