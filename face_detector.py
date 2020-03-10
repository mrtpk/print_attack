"""
Module to detect faces from an
RGB image using MTCNN.
"""

from mtcnn.mtcnn import MTCNN

class FaceDetector():
    def __init__(self):
        self.detector = MTCNN()

    def get_faces(self, img):
        """
        Returns bounding boxes of the
        faces in :param img
        """
        faces = self.detector.detect_faces(img)
        boxes = []
        for face in faces:
            x1, y1, width, height = face[0]['box']
            x1 = abs(x1)
            y1 = abs(y1)
            x2 = x1 + width
            y2 = y1 + height
            boxes.append([x1, y1, x2, y2])
        return boxes
