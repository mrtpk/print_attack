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

def detection_pipeline():
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
                # TODO: Adjust this threshold 
                person_name = face_recognizer.predict(embedding=embedding, threshold=0.8, verbose=True)
                print("[INFO]: Detected person is {}".format(person_name))
                # TODO: Adjust this threshold 
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

def create_dataset():
    import pathlib
    from glob import glob
    _path = "./data/collected_data/{}/{}/{}/" # fake/real, full/crop, name
    _img_name = "{}_{}.jpg" # name_id.jpg
    name = input("[INPUT]: Enter name of person: ")
    name = name.lower().strip()
    # create paths
    pathlib.Path(_path.format("fake", "full", name)).mkdir(parents=True, exist_ok=True)
    pathlib.Path(_path.format("real", "full", name)).mkdir(parents=True, exist_ok=True)
    pathlib.Path(_path.format("fake", "crop", name)).mkdir(parents=True, exist_ok=True)
    pathlib.Path(_path.format("real", "crop", name)).mkdir(parents=True, exist_ok=True)
    _counter_fake = len(list(glob(_path.format("fake", "full", name) + "*.jpg")))
    _counter_real = len(list(glob(_path.format("real", "full", name) + "*.jpg")))
    print("[INFO]: Collecting data for {}.".format(name))
    print("[INSTRUCTIONS]:\nPress `p` to pause frame")
    print("Press `r` to label the face as real.")
    print("Press `f` to label the face as fake.")
    print("Press `s` to save.")
    print("Press `k` to skip.")
    face_detector = FaceDetector()
    print("[INFO]: Starting camera stream")
    camera = CameraStream(0)
    frame = camera.read()
    if frame is None:
        print("[ERROR]: Cannot open camera. Quitting.")
        exit()
    
    COLOR_DETECTED_FACE = (255,0,0)
    COLOR_ATTACK_LABEL = {0:(0,255,0), 1: (0,0,255)}
    while(True):
        frame = camera.read()
        render = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        boxes = face_detector.get_faces(frame)
        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(render, (x1, y1), (x2, y2), COLOR_DETECTED_FACE, 1)
        cv2.imshow("Presentation attack detection: data collection", cv2.flip(render, 1))
        key_pressed = cv2.waitKey(3)
        if key_pressed == ord("q"):
            break
        if key_pressed == ord("p"):
            # process frame
            _label = ""
            _frame = frame[..., ::-1]
            _crop = _frame[y1:y2, x1:x2, :]
            print("[INFO]: Real or Fake?")
            print("Press `r` to label the face as real.")
            print("Press `f` to label the face as fake.")
            print("Press `k` to skip.")
            key_pressed = cv2.waitKey(0)
            if key_pressed == ord("k"):
                continue
            if key_pressed == ord("q"):
                break
            if key_pressed == ord("f"):
                _label = "fake"
                # _path = "./data/collected_data/{}/{}/{}/" # fake/real, full/crop, name
                # _img_name = "{}_{}.jpg" # name_id.jpg
                _counter_fake += 1
                _full_path = _path.format("fake", "full", name) + _img_name.format(name, _counter_fake)
                _crop_path = _path.format("fake", "crop", name) + _img_name.format(name, _counter_fake)
                color = COLOR_ATTACK_LABEL[1]
                cv2.rectangle(render, (x1, y1), (x2, y2), color, 1)
            elif key_pressed == ord("r"):
                _label = "real"
                _counter_real += 1
                _full_path = _path.format("real", "full", name) + _img_name.format(name, _counter_real)
                _crop_path = _path.format("real", "crop", name) + _img_name.format(name, _counter_real)
                color = COLOR_ATTACK_LABEL[0]
                cv2.rectangle(render, (x1, y1), (x2, y2), color, 1)
            else:
                print("[INFO]: Invalid label. Skipping.")
                continue
            cv2.imshow("Presentation attack detection: data collection", cv2.flip(render, 1))
            print("[INFO]: Save or Skip?")
            print("Press `s` to save.")
            print("Press `k` to skip.")
            key_pressed = cv2.waitKey(0)
            if key_pressed == ord("k"):
                continue
            if key_pressed == ord("q"):
                break
            if key_pressed == ord("s"):
                cv2.imwrite(_full_path, _frame)
                cv2.imwrite(_crop_path, _crop)
if __name__ == "__main__":
    # create_dataset()
    detection_pipeline()
