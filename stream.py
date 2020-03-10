"""
Module to stream video
"""
import cv2

class CameraStream():
    def __init__(self, camera):
        self.stream = cv2.VideoCapture(0)
        self.is_stop = False
    
    def read(self):
        """
        Read an image from camera and
        returns in RGB format.
        """
        if not self.is_stop:
            ret, frame = self.stream.read()
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None
    
    def stop(self):
        self.is_stop = True
        self.stream.release()
