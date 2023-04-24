import cv2
import time
import logging
class Camera:
    def __init__(self, id, res) -> None:
        '''
        id - 'ls /dev/video*'
        '''
        self.cam = cv2.VideoCapture(id)
        self.id = id
        self.window_name = f'camera_{id}_{res}'
        self.cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        cv2.VideoWriter()
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, res[0])
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, res[1])

        self.img_counter = 0
        for i in range(50):
            ret, frame = self.update()
        self._window_inited = False
        logging.basicConfig()
        logging.info(f'''init camera {self.id}
CAP_PROP_FPS            : {self.cam.get(cv2.CAP_PROP_FPS )}
CAP_PROP_FRAME_WIDTH    : {self.cam.get(cv2.CAP_PROP_FRAME_WIDTH  )}
CAP_PROP_FRAME_HEIGHT   : {self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT )}
CAP_PROP_FORMAT         : {self.cam.get(cv2.CAP_PROP_FORMAT )}
CAP_PROP_AUTO_EXPOSURE  : {self.cam.get(cv2.CAP_PROP_AUTO_EXPOSURE )}
CAP_PROP_EXPOSURE       : {self.cam.get(cv2.CAP_PROP_EXPOSURE  )}
CAP_PROP_AUTO_WB        : {self.cam.get(cv2.CAP_PROP_AUTO_WB )}
CAP_PROP_WB_TEMPERATURE : {self.cam.get(cv2.CAP_PROP_WB_TEMPERATURE )}
                    ''')



    def update(self):
        ret, frame = self.cam.read()
        self.img_counter += 1
        return ret, frame



    def end(self):
        self.cam.release()



    def init_window(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        ratio = 2
        cv2.resizeWindow(   self.window_name,
                            int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH)//ratio),
                            int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT)//ratio))
        self._window_inited = True

    def show(self, oneframe = False):
        raise DeprecationWarning()
        if not self._window_inited: self.init_window()
        while True:
            ret, frame = self.update()
            if not ret:
                logging.info("failed to grab frame")
                break
            cv2.imshow(self.window_name, frame)
            k = cv2.waitKey(1)
            if k%256 == 27:
                # ESC pressed
                logging.info("Escape hit, closing...")
                break
            elif k%256 == 32:
                # SPACE pressed
                img_name = f"opencv_frame_{self.img_counter}.png"
                cv2.imwrite(img_name, frame)
                logging.info("{} written!".format(img_name))
            if oneframe: return True
        cv2.destroyWindow(self.window_name)




class MockCamera(Camera):
    def __init__(self, id_, res) -> None:
        mock_filename = 'mock.mp4'
        import os
        self.file_name = f'{id(self)}_{mock_filename}'
        os.system(f'cp {mock_filename} {self.file_name}')
        super().__init__(self.file_name, res)

    
    def update(self):
        ret, frame = super().update()
        if not ret:
            self.cam.open(self.file_name)
            return super().update()
        return ret, frame
    
    def end(self):
        rt = super().end()
        import os
        os.remove(self.file_name)

        return rt

    def show(self, oneframe=False):
        ret = super().show(oneframe)
        if not ret:
            self.cam.open(self.file_name)
            return super().update()
        return ret
