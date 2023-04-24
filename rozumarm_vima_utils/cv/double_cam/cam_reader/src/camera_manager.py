# from camera import Camera
from .camera import Camera, MockCamera
import threading
import logging
import time
from typing import Iterable
import cv2


class CameraReaderThread(threading.Thread):
    def __init__(self, id, res) -> None:
        super().__init__()
        self.id = id
        self.res = res
        self.img = None
        self._stop_event = threading.Event()
        self._cam_inited = threading.Event()

    def run(self) -> None:
        cam = Camera(self.id, self.res)
        ret, self.img = cam.update()
        self._cam_inited.set()
        while ret and not self._stop_event.is_set():
            ret, self.img = cam.update()
    
        cam.end()
        # return super().run()
    
    def wait_camera(self):
        self._cam_inited.wait()

    def stop(self):
        self._stop_event.set()
    



class CameraManager:
    def __init__(self) -> None:
        self.camthreads = dict() 
        self.stop_event = threading.Event()
        self.show_thread = threading.Thread(target=self._showThread, name='vizualize')

    def add(self, id, res):
        '''
        id - int | Iterable[int]
        res - int | Iterable[int] | Iterable[[int,int]]       
        '''
        if isinstance(res, Iterable):
            if len(res) == 2 and isinstance(id, Iterable):
                res = [res]*len(id)
            if not isinstance(res[0], Iterable):
                res = [(r, r) for r in res] 
        elif isinstance(id, Iterable) and not isinstance(id, str):
            res = [(res, res)]*len(id)

        if isinstance(id, Iterable) and not isinstance(id, str):
            assert len(id) == len(res), f'lenght id neq res: {len(id)}!= {len(res)}'
            assert isinstance(res[0], Iterable), f'res is not Iterable'
            for cam_id, size in zip(id, res):
                self.camthreads[cam_id] = CameraReaderThread(id, size)
        else:
            res = res if isinstance(res, Iterable) else (res, res)
            self.camthreads[id] = CameraReaderThread(id, res)



    def _showThread(self):
        for name, cam in self.camthreads.items():
            print(name)
            cv2.startWindowThread()
            cv2.namedWindow(str(name), cv2.WINDOW_NORMAL)
            ratio = 2
            cv2.resizeWindow(   str(name),
                                int(cam.img.shape[1]//ratio),
                                int(cam.img.shape[0]//ratio))
        is_exit = False
        while not is_exit and not self.stop_event.isSet():
            for name, cam in self.camthreads.items(): 
                frame = cam.img
                cv2.imshow(str(name), frame)
                k = cv2.waitKey(1)
                if k%256 == 27:
                    # ESC pressed
                    logging.info("Escape hit, closing...")
                    is_exit = True
                    break
                elif k%256 == 32:
                    # SPACE pressed
                    img_name = f"opencv_frame_{cam.img_counter}.png"
                    cv2.imwrite(img_name, frame)
                    logging.info("{} written!".format(img_name))

        cv2.destroyAllWindows()


    def show(self):
        self.show_thread.start()

    def start(self):
        for key, v in self.camthreads.items():
            v.start()
        for key, v in self.camthreads.items():
            v.wait_camera()
        

    def stop(self):
        self.stop_event.set()
        for key, v in self.camthreads.items():
            v.stop()

        
    def get_image(self, id):
        return self.camthreads[id].img.copy()


def mock_test():
    global Camera
    Camera = MockCamera
    cam_manager = CameraManager()
    cam_manager.add(['/dev/video1', '/dev/video2'], 720)
    cam_manager.start()
    cam_manager.show()
    try:
        while True:
            time.sleep(1)
            logging.info('IM ALIVE')
    except KeyboardInterrupt:
        print('interrupted')
        cam_manager.stop()
    
def camera_test():
    global Camera
    from .camera import Camera
    cam_manager = CameraManager()
    cam_manager.add(0, 720)
    cam_manager.start()
    cam_manager.show()
    try:
        while True:
            time.sleep(1)
            logging.info('IM ALIVE')
    except KeyboardInterrupt:
        print('interrupted')
        cam_manager.stop()

if __name__ == '__main__':

    FORMAT = '[%(levelname)s][%(threadName)s/%(module)s/%(funcName)s]: %(message)s'
    logging.basicConfig(level = logging.DEBUG, format=FORMAT)
    mock_test()
    # camera_test()
