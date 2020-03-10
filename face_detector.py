"""
Module to detect faces from an
RGB image using MTCNN.
"""

from mtcnn.mtcnn import MTCNN

class FaceDetector():
    def __init__(self, threshold=0.9):
        self.detector = MTCNN()
        self.threshold = threshold

    def get_faces(self, img):
        """
        Returns bounding boxes of the
        faces in :param img
        """
        faces = self.detector.detect_faces(img)
        # print(faces)
        boxes = []
        for face in faces:
            if face['confidence'] < self.threshold:
                continue
            x1, y1, width, height = face['box']
            x1 = abs(x1)
            y1 = abs(y1)
            x2 = x1 + width
            y2 = y1 + height
            boxes.append([x1, y1, x2, y2])
        return boxes
