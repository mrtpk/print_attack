"""
Attack detection app
"""
import cv2
import numpy as numpy

from stream import CameraStream
from embedding import FaceEmbedder
from face_detector import FaceDetector
from face_recognition import FaceRecongizer
from inference import AttackDetector

if __name__ == "__main__":

    COLOR_DETECTED_FACE = (255,0,0)
    COLOR_ATTACK_LABEL = {0:(0,255,0), 1: (0,0,255)}

    print("[INFO]: Starting application")

    camera = CameraStream(0)
    face_detector = FaceDetector()
    face_recognizer = FaceRecongizer()

    embedder = FaceEmbedder()
    attack_detector = AttackDetector()

    frame = camera.read()
    if frame is None:
        print("[ERROR]: Cannot open camera. Quitting.")

    while(True):
        frame = camera.read()
        render = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        boxes = face_detector.get_faces(frame)
        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(render, (x1, y1), (x2, y2), COLOR_DETECTED_FACE, 1)
        cv2.imshow("Presentation attack detection", cv2.flip(render, 1))

        key_pressed = cv2.waitKey(3)
        if key_pressed == ord("d"):
            # check for presentation attack
            # this is done to avoid burden on system resources
            for x1, y1, x2, y2 in boxes:
                crop = frame[y1:y2, x1:x2, :]
                embedding = embedder.get_embedding(crop)
                person_name = face_recognizer.predict(embedding=embedding, threshold=0.8, verbose=True)
                # print("[INFO]: Detected person is {}".format(person_name))
                is_attack = attack_detector.is_attack(embedding=embedding, threshold=0.5, verbose=True)
                color = COLOR_ATTACK_LABEL[is_attack]
                cv2.rectangle(render, (x1, y1), (x2, y2), color, 1)
            
            cv2.imshow("Presentation attack detection", cv2.flip(render, 1))
            # wait till the user press another key
            key_pressed = cv2.waitKey(0)

        if key_pressed == ord("q"):
            break
    cv2.destroyAllWindows()
    print("[INFO]: Application closed")