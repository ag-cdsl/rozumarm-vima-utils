from typing import Optional

import numpy as np
import pybullet
from vima_bench import make
from scipy.spatial.transform import Rotation

from .transform import rf_tf_r2v, map_tf_repr


UR5_BODY_ID = 2
SPATULA_BODY_ID = 3


class VIMASceneRenderer:
    """
    top-view: x down, y right
    center of constrained zone at approx (0.45, 0)
    """

    def __init__(self, task_name: str, hide_arm_rgb=True):
        assert task_name in (
            "sweep_without_exceeding",
            "sweep_without_touching",
        ), "Non constraint-satisfaction tasks are not supported."
        self.env = make(
            task_name=task_name,
            modalities="rgb",
            # display_debug_window=True,
            hide_arm_rgb=hide_arm_rgb
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
        - assumes arm is hidden
        
        obj_posquats: iterable of (2d-position, quaternion) tuples,
            swept objects come first, distractors come last
        from_rozumarm_rf: whether posquats are given in rf of rozum-arm or VIMA
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

    def render_full_view(self):
        view_mat = pybullet.computeViewMatrix(
            cameraEyePosition=(10., 0, 8.),
            cameraTargetPosition=(0., 0., -0.01),
            cameraUpVector=(-0.707106, 0.0, 0.707106)
        )

        aspect_ratio = 16 / 9

        proj_mat = pybullet.computeProjectionMatrixFOV(
            fov=4.5,
            aspect=aspect_ratio,
            nearVal=9.,
            farVal=100.,
        )

        width = 1024

        w, h, image, *_ = pybullet.getCameraImage(
                width=width,
                height=int(width / aspect_ratio),
                viewMatrix=view_mat,
                projectionMatrix=proj_mat,
                flags=pybullet.ER_NO_SEGMENTATION_MASK,
                renderer=pybullet.ER_BULLET_HARDWARE_OPENGL,
                physicsClientId=self.env.client_id,
            )

        image = np.array(image, dtype=np.uint8).reshape((h, w, 4))
        image = image[:, :, :3]
        return image
    
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

    def set_arm_state(self, tcp_pos, tcp_quat):
        tcp_rot = Rotation.from_quat(tcp_quat)

        # shift from body COM
        bias = [0, -0.09, 0]
        bias = tcp_rot.apply(bias)
        target_pos = [a + b for a, b in zip(tcp_pos, bias)]

        joint_angles = pybullet.calculateInverseKinematics(
            bodyUniqueId=UR5_BODY_ID,
            endEffectorLinkIndex=7,  # solving for last link
            targetPosition=target_pos,
            targetOrientation=tcp_rot.as_quat(),
            jointDamping=[0.0001] * 6,
            solver=0,
            maxNumIterations=100,
            residualThreshold=0.001
        )

        # move joints
        for i, theta in zip(range(2, 7), joint_angles):
            pybullet.resetJointState(
                bodyUniqueId=UR5_BODY_ID,
                jointIndex=i,
                targetValue=theta,
                physicsClientId=self.env.client_id
            )
        
        # move eef
        last_link_state = pybullet.getLinkState(
            UR5_BODY_ID,
            7,
            physicsClientId=self.env.client_id
        )
        last_link_pos, last_link_quat, *_ = last_link_state

        last_link_rot = Rotation.from_quat(last_link_quat)
        eef_rot_wrt_last_link = Rotation.from_euler('xyz', (-90, 0, 0), degrees=True)
        tcp_rot = (last_link_rot * eef_rot_wrt_last_link).as_quat()
        
        bias = [0, 0.07, 0]
        bias = last_link_rot.apply(bias)
        eef_pos = [a + b for a, b in zip(last_link_pos, bias)]

        pybullet.resetBasePositionAndOrientation(
            bodyUniqueId=SPATULA_BODY_ID,
            posObj=eef_pos,
            ornObj=tcp_rot,
            physicsClientId=self.env.client_id,
        )
