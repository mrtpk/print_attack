from data_process import pickle_read
from data_generator import DataGenerator, load_all
from classifier_models import get_classifier
from sklearn.metrics import precision_recall_fscore_support
import numpy as np

from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense,Dropout,Softmax,Flatten,Activation,BatchNormalization
import tensorflow as tf

class FaceRecongizer():
    def __init__(self):
        self.labels = {-1: 'unknown', 0: 'ashwin', 1: 'tessa'} # TODO: change the labels
        self.load_pretrained = True
        self.path_save_model = "./data/models/face_recognizer/face_recognizer.h5"
        self.get_model()
            
    def get_model(self):
        if self.load_pretrained:
            self.model = load_model(self.path_save_model)
            return
        self.model = Sequential()	
        self.model.add(Dense(units=100,input_dim=128,kernel_initializer='glorot_uniform'))		
        self.model.add(BatchNormalization())		
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(units=10,kernel_initializer='glorot_uniform'))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(units=len(self.labels),kernel_initializer='he_uniform'))
        self.model.add(Activation('softmax'))
        self.model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),optimizer='nadam',metrics=['accuracy'])
    
    def train(self):
        # Load meta data of dataset
        train_set = pickle_read("./data/image_recognition/processed/train.pkl")
        valid_set = pickle_read("./data/image_recognition/processed/valid.pkl")
        test_set = pickle_read("./data/image_recognition/processed/test.pkl")
        self.get_model()
        # Train with data generator
        gen_train = DataGenerator(dataset=train_set, batch_size=8, shuffle=True)
        gen_valid = DataGenerator(dataset=valid_set, batch_size=8, shuffle=True)
        gen_test = DataGenerator(dataset=test_set, batch_size=8)
        history = self.model.fit_generator(epochs=3, generator=gen_train, validation_data=gen_valid)
        scores = self.model.evaluate_generator(gen_test, 4)
        print("scores", scores[1])
        # # Train without data generator 
        # x_train, y_train = load_all(train_set)
        # x_valid, y_valid = load_all(valid_set)
        # x_test, y_test = load_all(test_set)
        # history = self.model.fit(x=x_train, y=y_train, batch_size=128,
        #                     epochs=3, validation_data=(x_valid, y_valid), shuffle=True)
        # scores = self.model.evaluate(x=x_test, y=y_test, batch_size=128)

        self.model.save(self.path_save_model)

    def evaluate(self):
        test_set = pickle_read("./data/image_recognition/processed/test.pkl")
        # gen_test = DataGenerator(dataset=test_set, batch_size=4)
        # data generator alternate
        x_test, y_test = load_all(test_set)
        preds = self.model.predict(x_test)
        preds = np.argmax(preds, axis=-1)
        return precision_recall_fscore_support(y_true=y_test, y_pred=preds, average="micro")
    
    def predict(self, embedding, threshold, verbose):
        embedding = np.expand_dims(embedding, axis=0)
        pred = self.model.predict(embedding)
        if verbose:
            print("Recognition probability:", np.max(pred))
            print("Person:", self.labels[np.argmax(pred, axis=-1)[0]])
            print("Is valid:", np.max(pred) > threshold)
        if np.max(pred) > threshold:
            return self.labels[np.argmax(pred, axis=-1)[0]]
        return self.labels[-1]

if __name__ == "__main__":
    face_recognizer = FaceRecongizer()
    face_recognizer.train()
    print("precision, recall, fscore, support:", face_recognizer.evaluate())
