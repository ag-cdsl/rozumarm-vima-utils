from typing import Optional

import numpy as np
import pybullet
from vima_bench import make

from transform import rf_tf_r2v, map_tf_repr


class VIMASceneRenderer:
    """
    top-view: x down, y right
    center of constrained zone at approx (0.45, 0)
    """

    def __init__(self, task_name: str):
        assert task_name in (
            "sweep_without_exceeding",
            "sweep_without_touching",
        ), "Non constraint-satisfaction tasks are not supported."
        self.env = make(
            task_name=task_name,
            modalities="rgb",
        )

    def reset(self, exact_num_swept_objects: Optional[int] = None):
        """
        exact_num_swept_objects: the exact number of swept objects to use
        """
        if exact_num_swept_objects is not None:
            self._set_exact_num_swept_objs(exact_num_swept_objects)

        self.env.reset()

        self.swept_obj_ids = [x[0][0][0] for x in self.env.task.goals]
        self.distractor_obj_ids = list(self.env.task.distractors_pts.keys())
        self.all_obj_ids = self.swept_obj_ids + self.distractor_obj_ids
        self.n_swept_objs = len(self.swept_obj_ids)

        self.zs = []
        for obj_id in self.all_obj_ids:
            (*_, z), _ = pybullet.getBasePositionAndOrientation(
                bodyUniqueId=obj_id,
                physicsClientId=self.env.client_id
            )
            self.zs.append(z)

    def render_scene(self, obj_posquats, from_rozumarm_rf: bool = True):
        """
        obj_posquats: iterable of (2d-position, quaternion) tuples in VIMA's reference frame,
            swept objects come first, distractors come last
        from_rozumarm_rf: whether posquats are given in rozum arm rf
        """
        assert len(obj_posquats) == 2 * self.n_swept_objs, "Wrong number of positions."

        # map to rf
        if from_rozumarm_rf:
            obj_posquats = [
                (rf_tf_r2v(pos), map_tf_repr(quat)) for pos, quat in obj_posquats
            ]

        for (pos, quat), obj_id, z in zip(obj_posquats, self.all_obj_ids, self.zs):
            assert len(pos) == 2, "Specify 2d positions."
            pybullet.resetBasePositionAndOrientation(
                bodyUniqueId=obj_id,
                posObj=tuple(pos) + (z,),
                ornObj=quat,
                physicsClientId=self.env.client_id,
            )
        images = self._get_images()
        return images  # (front, top)

    def _get_images(self):
        obs = self.env._get_obs()
        images = [
            np.transpose(obs["rgb"][view], axes=(1, 2, 0)) for view in ("front", "top")
        ]
        return images

    def _set_exact_num_swept_objs(self, n: int):
        assert 1 <= n <= 3, "Number of swept objects should be in [1 .. 3]"
        self.env.task.task_meta["sample_prob"] = {
            i: float(i == n) for i in range(1, n + 1)
        }
