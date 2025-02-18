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

# --- v3: correct rot ---

# C2R_TF_SCALE = 0.98478
# C2R_T = Rotation.from_euler('XYZ',
#     [ 0.,  0., 90.5767351],
#     degrees=True
# ).inv()
# R2C_T = C2R_T.inv()
# C2R_B = np.array([-0.35048685, 0.00544445], dtype=np.float64)  # points from R to B, described in R


# --- v4: corrected rrf dataset --- (working!)

# C2R_TF_SCALE = 0.98478
# C2R_T = Rotation.from_euler('XYZ',
#     [ 0.,  0., 90.5767351],
#     degrees=True
# ).inv()
# R2C_T = C2R_T.inv()
# C2R_B = np.array([-0.35048685, 0.00094445], dtype=np.float64)  # points from R to B, described in R


# --- v5: removed scale correction

C2R_TF_SCALE = 1.0
C2R_T = Rotation.from_euler('XYZ',
    [ 0.,  0., 90.5767351],
    degrees=True
).inv()
R2C_T = C2R_T.inv()
C2R_B = np.array([-0.3514728, -0.00007647], dtype=np.float64)  # points from R to B, described in R

# --- v5.1: tmp test ---

# C2R_TF_SCALE = 1.0
# C2R_T = Rotation.from_euler('XYZ',
#     [ 0.,  0., 90.52583644],
#     degrees=True
# ).inv()
# R2C_T = C2R_T.inv()
# C2R_B = np.array([-3.51584573e-01,  1.74344281e-06], dtype=np.float64)  # points from R to B, described in R


def rf_tf_c2r(vec, apply_translation=True):
    v = C2R_TF_SCALE * C2R_T.apply((*vec, 0))[:2]
    if apply_translation:
        v += C2R_B
    return v


def map_tf_repr_c2r(quat):
    return (Rotation.from_quat(quat) * R2C_T).as_quat()


import numpy as np
from scipy.spatial.transform import Rotation

def make_transform_matrix(p, q):
    """Makes 4D transform matrix from tranlation vector and quaternion.
    
    returns: np.array (4, 4)
    """
    rotation_matrix = Rotation.from_quat(q).as_matrix()
    res = np.vstack((np.hstack((rotation_matrix, p[:, None])), [0, 0, 0, 1]))
    return res


def get_transform_between_frames(p1, q1, p2, q2):
    """Computes transform between RF described by (p1, a1) and another described by (p2, q2)
    
    returns: 4D matrix
    
    t1 describes 0 -> 1
    t2 describes 0 -> 2
    1 -> 2 is described by t1.inv() * t2
    """
    tf_01 = make_transform_matrix(p1, q1)
    tf_02 = make_transform_matrix(p2, q2)
    tf_12 = np.linalg.inv(tf_01) @ tf_02
    return tf_12
