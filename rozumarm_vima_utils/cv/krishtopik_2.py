import numpy as np
import cv2


def calibrate_table(image, K, D):
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_1000)
    aruco_params = cv2.aruco.DetectorParameters_create()
    aruco_params.adaptiveThreshConstant = 23
    marker_corners, _ = detect_aruco(image, K, D, 0.125, True, aruco_dict, aruco_params)
    assert len(marker_corners) == 16, str(len(marker_corners))
    camera2table = estimate_plane(marker_corners)

    correction = np.eye(4)
    correction[0:3, 0:3], _ = cv2.Rodrigues(np.array([0., 0., 1.]) * 0.4167962463787207)
    camera2table = np.matmul(camera2table, correction)
    return camera2table


def detect_aruco(image, K, D, aruco_sizes, extract_all_corners,
        aruco_dict, aruco_params):
    corners, ids, _ = \
        cv2.aruco.detectMarkers(image, aruco_dict, parameters=aruco_params)
    corners = np.array(corners)
    n = corners.shape[0]
    # corners.shape = (n, 1, 4, 2)
    # ids.shape = (n, 1)

    ind = np.argsort(ids, axis=0)
    ids = np.take_along_axis(ids, ind, axis=0)
    corners = np.take_along_axis(corners, np.expand_dims(ind, axis=(-1, -2)), axis=0)

    if not isinstance(aruco_sizes, (list, tuple, np.ndarray)):
        aruco_sizes = np.array([aruco_sizes] * n)
    if len(aruco_sizes.shape) != 1:
        raise RuntimeError(f"Use list, tuple or np.ndarray to pass multiple aruco sizes.")
    if aruco_sizes.shape != (n,):
        raise RuntimeError(
            f"Number of aruco marker sizes does not correspond to "
            f"the number of detected markers ({aruco_sizes.shape[0]} vs {n})")

    rvecs = list()
    tvecs = list()
    for i in range(n):
        rvec, tvec, _ = \
            cv2.aruco.estimatePoseSingleMarkers(corners[i], aruco_sizes[i], K, D)
        rvecs.append(rvec)
        tvecs.append(tvec)
    rvecs = np.array(rvecs)
    tvecs = np.array(tvecs)
    # rvecs.shape = (n, 1, 3)
    # tvecs.shape = (n, 1, 3)

    marker_poses = np.tile(np.eye(4), (n, 1, 1))
    for i in range(n):
        marker_poses[i, 0:3, 0:3], _ = cv2.Rodrigues(rvecs[i])
        marker_poses[i, 0:3, 3] = tvecs[i, 0]
    # marker_poses.shape = (n, 4, 4)

    corners_in_marker_frames = list()
    for i in range(n):
        corners_in_single_marker_frame = list()
        for sx, sy in [(-1, 1), (1, 1), (1, -1), (-1, -1)]:
            # top left corner first
            single_corner_in_marker_frame = np.array(
                [aruco_sizes[i] / 2 * sx,
                 aruco_sizes[i] / 2 * sy,
                 0, 1]).reshape(-1, 1)
            corners_in_single_marker_frame.append(single_corner_in_marker_frame)
            if not extract_all_corners:
                break
        corners_in_single_marker_frame = np.array(corners_in_single_marker_frame)
        corners_in_marker_frames.append(corners_in_single_marker_frame)
    corners_in_marker_frames = np.array(corners_in_marker_frames).swapaxes(0, 1)
    # corners_in_marker_frames.shape = (1 or 4, n, 4, 1)

    marker_corners = np.matmul(marker_poses, corners_in_marker_frames)
    marker_corners = marker_corners[:, :, 0:3, 0].swapaxes(0, 1).reshape(-1, 3)
    # marker_corners.shape = (n or n * 4, 3)

    return marker_corners, \
        {'n': n, 'corners': corners, 'ids': ids, 'rvecs': rvecs, 'tvecs': tvecs}


def project_to_plane(p0, plane):
    x0 = p0[0]
    y0 = p0[1]
    z0 = p0[2]
    a = plane[0]
    b = plane[1]
    c = plane[2]

    dz = (a * x0 + b * y0 + c - z0) / (a * a + b * b + 1)
    x = x0 - a * dz
    y = y0 - b * dz
    z = z0 + dz

    return np.array([x, y, z])


def estimate_plane(points):
    # ax + by + c = z

    # A * plane = B
    # plane = [a, b, c]
    # A = [[xi, yi, 1]]
    # B = [zi]

    # plane = inv(AT * A) * AT * B

    n = points.shape[0]
    A = np.hstack((points[:, 0:2], np.ones((n, 1))))
    B = points[:, 2]
    plane = np.matmul(np.matmul(np.linalg.inv(np.matmul(A.T, A)), A.T), B)

    centroid = np.sum(points, axis=0) / n
    origin = project_to_plane(centroid, plane)

    a = plane[0]
    b = plane[1]
    c = plane[2]
    z_axis = np.array([-a, -b, 1])
    z_axis /= np.linalg.norm(z_axis)
    x_axis = project_to_plane(points[0], plane) - origin
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    if np.dot(z_axis, origin) > 0:
        y_axis *= -1
        z_axis *= -1

    x_axis = np.expand_dims(x_axis, axis=-1)
    y_axis = np.expand_dims(y_axis, axis=-1)
    z_axis = np.expand_dims(z_axis, axis=-1)

    R = np.hstack((x_axis, y_axis, z_axis))
    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = origin
    return T