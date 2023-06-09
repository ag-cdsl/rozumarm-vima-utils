from .camera import Camera, CamDenseReader
import numpy as np
import cv2

def toTuple(a, b):
    return [((*ai,), (*bi,)) for ai, bi in zip(a, b) ]

from detect_boxes import detect_boxes, detect_boxes_segm, detect_boxes_visual
from calibrate_table import calibrate_table_by_aruco, calibrate_table_by_markers
from params import table_aruco_size, box_aruco_size, box_size, K, D, target_table_markers



# def detect_cubes():
#     global calibrate_matrix
#     _, image = cam.update() 
#     if calibrate_matrix is None:
#         import time
#         start_t = time.time()
#         while time.time() - start_t < 5:
#             _, image = cam.update() 
#         calibrate_matrix = calibrate_table_by_aruco(image, K, D)
#     boxes_positions, boxes_orientations = detect_boxes(image, K, D, calibrate_matrix)
#     return toTuple(boxes_positions, boxes_orientations)

class CubeDetector():
    def __init__(self) -> None:
        self.cam = Camera(2, (1280, 1024))
        ret, image = self.cam.update()
        self.table_frame, _ = calibrate_table_by_aruco(image, "top", K, D, table_aruco_size)

    
    def detect(self):
        ret, image = self.cam.update()
        print(image.shape)
        cv2.imwrite('./test.jpg', image)
        boxes_positions, boxes_orientations = detect_boxes(image, "top", K, D, self.table_frame, box_aruco_size, box_size)
        print(f"Detected {len(boxes_positions)} boxes")
        res = toTuple( boxes_positions, boxes_orientations)
        for i, j in enumerate(res):
            print(f'cube #{i}: {j}')
        return res


import time
class CubeDenseDetector():
    def __init__(self) -> None:
        self.cam_1 = CamDenseReader(2, 'cam_top_video.mp4')
        self.cam_2 = CamDenseReader(0, 'cam_front_video.mp4')
        self.cam_1.start_recording()
        self.cam_2.start_recording()

        ret, image = self.cam_1.read_image()
        self.table_frame, _ = calibrate_table_by_aruco(image, "top", K, D, table_aruco_size)
        # self.table_transform = calibrate_table_by_markers(image, "top", K, D, target_table_markers)
        time.sleep(3)

    def detect(self):
        ret, image = self.cam_1.read_image()
        print(image.shape)
        cv2.imwrite('./test.jpg', image)
        # boxes_positions, boxes_orientations = detect_boxes(image, "top", K, D, self.table_frame, box_aruco_size, box_size)
        boxes_positions, boxes_orientations = detect_boxes_segm(image, "top", K, D, self.table_frame, box_size)
        # boxes_positions, boxes_orientations = detect_boxes_visual(image, "top", K, D, self.table_transform)
        print(f"Detected {len(boxes_positions)} boxes")
        res = toTuple( boxes_positions, boxes_orientations)
        for i, j in enumerate(res):
            print(f'cube #{i}: {j}')
        return res

    def release(self):
        self.cam_1.stop_recording()
        self.cam_2.stop_recording()


# detector = CubeDetector()
detector = CubeDenseDetector()

