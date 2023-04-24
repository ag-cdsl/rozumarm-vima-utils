import numpy as np
from scipy.spatial.transform import Rotation

# Rozum-Vima transforms
R2V_TF_SCALE = 1.0
# R2V_CUSTOM_BIAS = 0.3
R2V_CUSTOM_BIAS = 0.165

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
        res = T.apply(vec)
    else:
        res = T.apply((*vec, 0))[:2]
    
    res *= R2V_TF_SCALE
    res[0] += R2V_CUSTOM_BIAS
    return res


def rf_tf_v2r(vec):
    """Vima env to Rozum arm transform

    vec: of shape (2,)
    """
    vec[0] -= R2V_CUSTOM_BIAS
    return 1 / R2V_TF_SCALE * T.apply((*vec, 0))[:2]


def map_tf_repr(quat):
    return (Rotation.from_quat(quat) * T).as_quat()


# rotates (0, pi, 0)
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
    return (rot * GRIPPER_RF_ROT).as_quat()  # compose intrinsically


# Cam-Rozum transforms

# --- v0: before April 10 ---
# C2R_TF_SCALE = 1.022
# C2R_T = Rotation.from_matrix(
#     np.array([
#         [0, 1, 0],
#         [-1, 0, 0],
#         [0, 0, 1]
#     ], dtype=np.float64))
# R2C_T = C2R_T.inv()
# C2R_B = np.array([-0.27181241, -0.08476032], dtype=np.float64)  # points from R to B, described in R

# --- v1: after April 10 ---
C2R_TF_SCALE = 0.98
C2R_T = Rotation.from_matrix(
    np.array([
        [0, 1, 0],
        [-1, 0, 0],
        [0, 0, 1]
    ], dtype=np.float64))
R2C_T = C2R_T.inv()
C2R_B = np.array([-0.34974726, 0.00527472], dtype=np.float64)  # points from R to B, described in R


def rf_tf_c2r(vec, apply_bias=True):
    v = C2R_TF_SCALE * C2R_T.apply((*vec, 0))[:2]
    if apply_bias:
        v += C2R_B
    return v


def map_tf_repr_c2r(quat):
    return (Rotation.from_quat(quat) * R2C_T).as_quat()
