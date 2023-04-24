from .camera import Camera, CamDenseReader
import numpy as np
import cv2

def toTuple(a, b):
    return [((*ai,), (*bi,)) for ai, bi in zip(a, b) ]

from detect_boxes import detect_boxes
from calibrate_table import calibrate_table



# def detect_cubes():
#     global calibrate_matrix
#     _, image = cam.update() 
#     if calibrate_matrix is None:
#         import time
#         start_t = time.time()
#         while time.time() - start_t < 5:
#             _, image = cam.update() 
#         calibrate_matrix = calibrate_table(image, K, D)
#     boxes_positions, boxes_orientations = detect_boxes(image, K, D, calibrate_matrix)
#     return toTuple(boxes_positions, boxes_orientations)

class CubeDetector():
    def __init__(self) -> None:
        self.cam = Camera(2, (1280, 1024))
        self.cam_calibration = np.load('/home/daniil/Загрузки/rozumarm-vima-utils/rozumarm_vima_utils/scripts/calib.npz')
        self.K = self.cam_calibration['K']
        self.D = self.cam_calibration['D']
        ret, image = self.cam.update()
        self.calibrate_matrix, _ = calibrate_table(image, self.K, self.D, 0.132)

    
    def detect(self):
        ret, image = self.cam.update()
        # print(image.shape)
        # cv2.imwrite('./test.jpg', image)
        boxes_positions, boxes_orientations = detect_boxes(image, self.K, self.D, self.calibrate_matrix, 0.0172, 0.03)
        res = toTuple( boxes_positions, boxes_orientations)
        for i, j in enumerate(res):
            print(f'cube #{i}: {j}')
        return res


import time
class CubeDenseDetector():
    def __init__(self) -> None:
        self.cam_1 = CamDenseReader(2, 'cam_1_video.mp4')
        self.cam_2 = CamDenseReader(4, 'cam_2_video.mp4')
        self.cam_1.start_recording()
        self.cam_2.start_recording()

        self.cam_calibration = np.load('/home/daniil/Загрузки/rozumarm-vima-utils/rozumarm_vima_utils/scripts/calib.npz')
        self.K = self.cam_calibration['K']
        self.D = self.cam_calibration['D']
        ret, image = self.cam_1.read_image()
        self.calibrate_matrix, _ = calibrate_table(image, self.K, self.D, 0.132)
        time.sleep(3)
        

    
    def detect(self):
        ret, image = self.cam_1.read_image()
        # print(image.shape)
        # cv2.imwrite('./test.jpg', image)
        boxes_positions, boxes_orientations = detect_boxes(image, self.K, self.D, self.calibrate_matrix, 0.0172, 0.03)
        res = toTuple( boxes_positions, boxes_orientations)
        for i, j in enumerate(res):
            print(f'cube #{i}: {j}')
        return res

    def release(self):
        self.cam_1.stop_recording()
        self.cam_2.stop_recording()


detector = CubeDetector()
# detector = CubeDenseDetector()

