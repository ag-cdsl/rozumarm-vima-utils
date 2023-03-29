import numpy as np
from scipy.spatial.transform import Rotation

# Rozum-Vima transforms
R2V_TF_SCALE = 1.0

T = Rotation.from_matrix(
    np.array([[-1, 0, 0],
              [0, -1, 0],
              [0, 0, 1]], dtype=np.float64)
)


def rf_tf_r2v(vec, from3d=False):
    """Rozum arm to Vima env transform

    vec: of shape (2,)
    """
    if from3d:
        res = R2V_TF_SCALE * T.apply(vec)
        res[0] += 0.3
        return res
    res = R2V_TF_SCALE * T.apply((*vec, 0))[:2]
    res[0] += 0.3
    return res


def rf_tf_v2r(vec):
    """Vima env to Rozum arm transform

    vec: of shape (2,)
    """
    vec[0] -= 0.3
    return 1 / R2V_TF_SCALE * T.apply((*vec, 0))[:2]


def map_tf_repr(quat):
    return (Rotation.from_quat(quat) * T).as_quat()


GRIPPER_RF_ROT = Rotation.from_matrix(
        np.array([
        [-1, 0, 0],
        [0, 1, 0],
        [0, 0, -1]
    ])
)

def map_gripper_rf(quat):
    """
    active
    involutary
    """
    rot = Rotation.from_quat(quat)
    return (GRIPPER_RF_ROT * rot).as_quat()


# Cam-Rozum transforms
C2R_TF_SCALE = 1.022

C2R_T = Rotation.from_matrix(
    np.array([
        [0, 1, 0],
        [-1, 0, 0],
        [0, 0, 1]
    ], dtype=np.float64))
R2C_T = C2R_T.inv()
C2R_B = np.array([-0.27181241, -0.08476032], dtype=np.float64)  # points from R to B, described in R


def rf_tf_c2r(vec, apply_bias=True):
    v = C2R_TF_SCALE * C2R_T.apply((*vec, 0))[:2]
    if apply_bias:
        v += C2R_B
    return v


def map_tf_repr_c2r(quat):
    return (Rotation.from_quat(quat) * R2C_T).as_quat()
