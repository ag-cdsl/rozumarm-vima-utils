from .camera import Camera
import numpy as np

def toTuple(a, b):
    return [((*ai,), (*bi,)) for ai, bi in zip(a, b) ]

from .krishtopik_1 import detect_boxes
from .krishtopik_2 import calibrate_table



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
        self.cam_calibration = np.load('/home/daniil/Загрузки/rozumarm-vima-utils/rozumarm_vima_utils/scripts/camera_calibration.npz')
        self.K = self.cam_calibration['K']
        self.D = self.cam_calibration['D']
        ret, image = self.cam.update()
        self.calibrate_matrix = calibrate_table(image, self.K, self.D)

    
    def detect(self):
        ret, image = self.cam.update()
        boxes_positions, boxes_orientations = detect_boxes(image, self.K, self.D, self.calibrate_matrix)
        res = toTuple( boxes_positions, boxes_orientations)
        for i, j in enumerate(res):
            print(f'cube #{i}: {j}')
        return res

detector = CubeDetector()

