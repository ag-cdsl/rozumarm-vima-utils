import numpy as np
import cv2
from transforms3d.quaternions import axangle2quat
from transforms3d.axangles import mat2axangle


aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_1000)
aruco_params = cv2.aruco.DetectorParameters_create()
aruco_params.adaptiveThreshConstant = 15


def detect_boxes(image, K, D, camera2table, aruco_size=0.016, box_size=0.03):
    corners, ids, _ = \
        cv2.aruco.detectMarkers(image, aruco_dict, parameters=aruco_params)
    corners = np.array(corners)
    n = corners.shape[0]
    # corners.shape = (n, 1, 4, 2)
    # ids.shape = (n, 1)

    ind = np.argsort(ids, axis=0)
    ids = np.take_along_axis(ids, ind, axis=0)
    corners = np.take_along_axis(corners, np.expand_dims(ind, axis=(-1, -2)), axis=0)

    rvecs = list()
    tvecs = list()
    for i in range(n):
        rvec, tvec, _ = \
            cv2.aruco.estimatePoseSingleMarkers(corners[i], aruco_size, K, D)
        rvecs.append(rvec)
        tvecs.append(tvec)
    rvecs = np.array(rvecs)
    tvecs = np.array(tvecs)
    # rvecs.shape = (n, 1, 3)
    # tvecs.shape = (n, 1, 3)

    marker_poses_in_camera = np.tile(np.eye(4), (n, 1, 1))
    for i in range(n):
        marker_poses_in_camera[i, 0:3, 0:3], _ = cv2.Rodrigues(rvecs[i])
        marker_poses_in_camera[i, 0:3, 3] = tvecs[i, 0]
    # marker_poses_in_camera.shape = (n, 4, 4)

    marker_poses = np.matmul(np.linalg.inv(camera2table), marker_poses_in_camera)
    # marker_poses.shape = (n, 4, 4)

    marker2box = np.eye(4)
    marker2box[2, 3] = -box_size / 2
    boxes_poses = np.matmul(marker_poses, marker2box)
    # boxes_poses.shape = (n, 4, 4)

    boxes_positions = boxes_poses[:, 0:2, 3]
    # boxes_positions.shape = (n, 2)

    boxes_orientations = list()
    for i in range(n):
        axis, angle = mat2axangle(boxes_poses[i, 0:3, 0:3])
        assert abs(np.linalg.norm(axis) - 1.0) < 0.0001
        newAxis = np.array([0., 0., 1.])
        newAngle = angle * np.dot(axis, newAxis)
        quat = axangle2quat(newAxis, newAngle)
        quat = np.array([quat[1], quat[2], quat[3], quat[0]])
        boxes_orientations.append(quat)
    boxes_orientations = np.array(boxes_orientations)
    # boxes_orientations.shape = (n, 4)

    return boxes_positions, boxes_orientations