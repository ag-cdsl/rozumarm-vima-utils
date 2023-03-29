import cv2
import time
class Camera:
    def __init__(self, id : int, res) -> None:
        '''
        id - 'ls /dev/video*'
        '''
        self.cam = cv2.VideoCapture()
        self.cam.open(id)
        self.id = id 
        self.window_name = f'camera_{id}_{res}'
        self.cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, res[0])
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, res[1])
        # self.cam.set(cv2.CAP_PROP_AUTO_WB, 0.0)
        # self.cam.set(cv2.CAP_PROP_AUTO_EXPOSURE , 1000.0)
        # self.cam.set(cv2.CAP_PROP_EXPOSURE, 10)
        self.img_counter = 0
        for i in range(50):
            ret, frame = self.update()

    def show(self):
        cv2.namedWindow('test', cv2.WINDOW_NORMAL)
        cv2.resizeWindow("test", 300, 300)
        while True:
            ret, frame = self.update()
            if not ret:
                print("failed to grab frame")
                break
            cv2.imshow("test", frame)
            k = cv2.waitKey(1)
            if k%256 == 27:
                # ESC pressed
                print("Escape hit, closing...")
                break
            elif k%256 == 32:
                # SPACE pressed
                img_name = f"opencv_frame_{self.img_counter}.png"
                cv2.imwrite(img_name, frame)
                print("{} written!".format(img_name))
        cv2.destroyAllWindows()


    def update(self):
        ret, frame = self.cam.read()
        self.img_counter += 1
        return ret, frame
    
    def end(self):
        self.cam.release()