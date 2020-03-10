"""
Face embedding module. Converts face images of
size (160, 160) in RGB colorspace to a vector 
of 128 using FaceNet.
"""

import numpy as np
import cv2
import tensorflow as tf

class FaceEmbedder():
    """
    A face embedder using FaceNet.
    """
    def __init__(self, path_model='./data/models/facenet_keras.h5'):
        """
        Initialises embedder with FaceNet model.
        """
        self.path_model = path_model
        self.model = tf.keras.models.load_model(self.path_model, compile=False)
    
    def preprocess(self, img):
        """
        Preprocess the image for FaceNet.
        """
        img = cv2.resize(img, (160, 160), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32)
        _mean = img.mean()
        _std = img.std()
        img = (img - _mean) / _std # zero centering
        return img
    
    def get_embedding(self, img):
        """
        Returns FaceNet embedding for a given face.
        Expects the image to be in RGB color space.
        """
        img = self.preprocess(img) # preprocess for FaceNet model
        N_img = np.expand_dims(img, axis=0) # create a batch of image
        return self.model.predict(N_img)[0]

if __name__ == "__main__":
    embedder = FaceEmbedder()
    from scipy import spatial
    img = cv2.imread("./data/raw/ClientFace/0001/0001_00_00_01_0.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img2 = cv2.imread("./data/raw/ImposterFace/0001/0001_00_00_01_0.jpg")
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    result = 1 - spatial.distance.cosine(embedder.get_embedding(img), embedder.get_embedding(img2))
    print("Cosine similarity between real and fake is {}".format(result))