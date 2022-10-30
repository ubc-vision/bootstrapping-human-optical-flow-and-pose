import cv2
import numpy as np
class OpenCVCapture:
    def __init__(self, video_file=None):
        if video_file is None:
            self.cap = cv2.VideoCapture(int(args.cam_id))
        else:
            self.cap = cv2.VideoCapture(video_file)

    def read(self):
        flag, frame = self.cap.read()
        if not flag:
          return None
        return np.flip(frame, -1).copy() # BGR to RGB