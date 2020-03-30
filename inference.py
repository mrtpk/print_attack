"""
Module for doing inference using
trained classifiers.
"""
import numpy as np
from classifier_models import get_classifier
from data_process import pickle_read
from data_generator import DataGenerator, load_all
from classifier_models import get_classifier
from sklearn.metrics import precision_recall_fscore_support
from joblib import load
# TODO: modify this detector to accomodate SVM and SVC with kmeans
class AttackDetector():
    """
    To do detect attack by ensemble
    classfiers.
    """
    def __init__(self):
        self.classifiers = ["tiny_2_layer", "2_layer_dense"] # which classifiers to use
        self.ml_model_names = ["svm_liner", "gmm_spherical", "logistic_regression"]
        self.weights = [1 , 1, 1, 1, 1] # weights of each classifer
        assert len(self.classifiers + self.ml_model_names) == len(self.weights)
        self.weights = np.array(self.weights)
        self.models = []
        self.init_models()

    def init_models(self):
        for name in self.classifiers:
            classifier = get_classifier(name=name, load_pretrained=True)
            self.models.append(classifier)

        for name in self.ml_model_names:
            classifier = self.get_ml_model(name)
            self.models.append(classifier)

    def get_ml_model(self, name):
        path_save_model = "./data/models/{}_classifier/{}.joblib"
        return load(path_save_model.format(name.split("_")[0], name))

    def is_attack(self, embedding, threshold=0.9, verbose=False):
        embedding = np.expand_dims(embedding, axis=0)
        preds = []
        for model in self.models:
            pred = model.predict(embedding)
            pred = pred[0]
            if type(pred) == list:
                pred = pred[0]
            preds.append(pred)
        weighted_preds = np.array(preds) * self.weights
        weighted_preds = np.mean(weighted_preds)
        if verbose:
            print("Classfiers: ", self.classifiers)
            print("Attack Probability:", preds)
            print("Weights:", self.weights)
            print("Mean weight", weighted_preds)
            print("Is attack:", weighted_preds > threshold)
        if weighted_preds > threshold:
            return 1 # attack confirmed
        return 0 # not an attack

    def evaluate(self, threshold):
        """
        Returns Precision, Recall and FScore of
        the attack detector on test data.
        """
        test_set = pickle_read("./data/print_attack/processed/test.pkl")
        x_test, y_test = load_all(test_set)
        preds = []
        for x in x_test:
            pred = self.is_attack(x, threshold=threshold)
            preds.append(pred)
        # print(set(y_test) - set(preds))
        preds = np.array(preds)
        return precision_recall_fscore_support(y_true=y_test, y_pred=preds, average="binary")
