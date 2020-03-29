from data_process import pickle_read
from data_generator import DataGenerator, load_all
from classifier_models import get_classifier
from sklearn.metrics import precision_recall_fscore_support
import numpy as np

def evaluate_classifier(classifier, threshold):
    test_set = pickle_read("./data/print_attack/processed/test.pkl")
    # gen_test = DataGenerator(dataset=test_set, batch_size=128)
    # data generator alternate
    x_test, y_test = load_all(test_set)
    preds = classifier.predict(x_test)
    preds = ( preds > 0.9 ).astype(np.int64).squeeze()
    return precision_recall_fscore_support(y_true=y_test, y_pred=preds, average="binary")

if __name__ == "__main__":
    # Load meta data of dataset
    train_set = pickle_read("./data/print_attack/processed/train.pkl")
    valid_set = pickle_read("./data/print_attack/processed/valid.pkl")
    test_set = pickle_read("./data/print_attack/processed/test.pkl")
    path_save_model = "./data/models/attack_classifier/{}.h5"

    # Load classifier
    classifier_name = "2_layer_dense" # "tiny_2_layer", "2_layer_dense"
    model = get_classifier(classifier_name, load_pretrained=True)
    model.summary()
 
    # Train with data generator
    gen_train = DataGenerator(dataset=train_set, batch_size=128, shuffle=True)
    gen_valid = DataGenerator(dataset=valid_set, batch_size=128, shuffle=True)
    gen_test = DataGenerator(dataset=test_set, batch_size=128)
    history = model.fit_generator(epochs=100, generator=gen_train, validation_data=gen_valid)
    scores = model.evaluate_generator(gen_test, 128)

    # # Train without data generator 
    # x_train, y_train = load_all(train_set)
    # x_valid, y_valid = load_all(valid_set)
    # x_test, y_test = load_all(test_set)
    # model = get_classifier_model()
    # history = model.fit(x=x_train, y=y_train, batch_size=128,
    #                     epochs=3, validation_data=(x_valid, y_valid), shuffle=True)
    # scores = model.evaluate(x=x_test, y=y_test, batch_size=128)

    model.save(path_save_model.format(classifier_name))
    print("Accuracy = ", scores[1])
    print("precision, recall, fscore, support:", evaluate_classifier(classifier=model, threshold=0.9))

