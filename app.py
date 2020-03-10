"""
Attack detection app
"""
import cv2
import numpy as numpy

from embedding import FaceEmbedder
from face_detector import FaceDetector
from inference import AttackDetector

if __name__ == "__main__":
    print("Starting application")
