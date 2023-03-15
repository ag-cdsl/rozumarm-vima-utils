import numpy as np
from scipy.spatial.transform import Rotation


R2V_TF_SCALE = 1.0

T = Rotation.from_matrix(
    np.array([[-1, 0, 0],
              [0, -1, 0],
              [0, 0, 1]], dtype=np.float64)
)


def rf_tf_r2v(vec):
    """Rozum arm to Vima env transform

    vec: of shape (2,)
    """
    return R2V_TF_SCALE * T.apply((*vec, 0))[:2]


def rf_tf_v2r(vec):
    """Vima env to Rozum arm transform

    vec: of shape (2,)
    """
    return 1 / R2V_TF_SCALE * T.apply((*vec, 0))[:2]


def map_tf_repr(quat):
    return (Rotation.from_quat(quat) * T).as_quat()
