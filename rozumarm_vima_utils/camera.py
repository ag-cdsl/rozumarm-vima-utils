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
        # self.cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, res[0])
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, res[1])
        # self.cam.set(cv2.CAP_PROP_AUTO_WB, 0.0)
        # self.cam.set(cv2.CAP_PROP_AUTO_EXPOSURE , 1000.0)
        # self.cam.set(cv2.CAP_PROP_EXPOSURE, 10)
        self.img_counter = 0
        for i in range(50):
            self.cam.grab()
            # ret, frame = self.update()

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
        for _ in range(4):
            self.cam.grab()
        ret, frame = self.cam.read()
        self.img_counter += 1
        return ret, frame
    
    def end(self):
        self.cam.release()


import threading


def reader_thread(device_id, buf: list, is_reading: threading.Event, reading_done: threading.Event):
    cam = Camera(f'/dev/video{device_id}', (1280, 1024))

    while True:
        is_reading.wait()
        is_ok, image = cam.update()
        buf[0] = is_ok
        buf[1] = image
        is_reading.clear()
        reading_done.set()


class CamReader:
    def __init__(self, device_id: int):
        self.is_reading = threading.Event()
        self.reading_done = threading.Event()
        self.buf = [None, None]

        self.thread = threading.Thread(
            target=reader_thread,
            args=(device_id, self.buf, self.is_reading, self.reading_done)
        )
        self.thread.start()

    def read_image(self):
        self.is_reading.set()
        self.reading_done.wait()
        self.reading_done.clear()
        is_ok = self.buf[0]
        assert is_ok, "Failed reading image."
        image = self.buf[1]
        return is_ok, image


def dense_reader_thread(device_id, buf: list, is_reading: threading.Event, reading_done: threading.Event,
                        output_filename,
                        is_recording: threading.Event):
    cam = Camera(f'/dev/video{device_id}', (1280, 1024))
    writer = cv2.VideoWriter(
        output_filename,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps=30,
        frameSize=(1280, 1024)
    )

    while is_recording.is_set():
        # print('writer is opened, reading image')
        if is_reading.is_set():
            is_ok, image = cam.update()
            buf[0] = is_ok
            buf[1] = image
            is_reading.clear()
            reading_done.set()
        else:
            is_ok, image = cam.cam.read()
        writer.write(image)
    writer.release()
    cam.cam.release()
    print('writer is closed, exiting thread...')


class CamDenseReader:
    def __init__(self, device_id: int, output_filename: str):
        self.is_reading = threading.Event()
        self.reading_done = threading.Event()
        self.is_recording = threading.Event()
        self.is_recording.set()

        self.buf = [None, None]

        self.thread = threading.Thread(
            target=dense_reader_thread,
            args=(device_id, self.buf, self.is_reading, self.reading_done,
                  output_filename,
                  self.is_recording),
            daemon=False
        )

    def read_image(self):
        self.is_reading.set()
        self.reading_done.wait()
        self.reading_done.clear()
        is_ok = self.buf[0]
        assert is_ok, "Failed reading image."
        image = self.buf[1]
        return is_ok, image
    
    def start_recording(self):
        self.thread.start()
    
    def stop_recording(self):
        self.is_recording.clear()
