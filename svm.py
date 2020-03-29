from data_process import pickle_read
from data_generator import DataGenerator, load_all
from joblib import dump, load
import numpy as np
# dump(clf, 'filename.joblib') 
# clf = load('filename.joblib')
from sklearn import svm
from sklearn import metrics

if __name__ == "__main__":
    train_set = pickle_read("./data/print_attack/processed/train.pkl")
    valid_set = pickle_read("./data/print_attack/processed/valid.pkl")
    test_set = pickle_read("./data/print_attack/processed/test.pkl")
    name = "svm_liner"
    path_save_model = "./data/models/{}_classifier/{}.joblib"
    x_train, y_train = load_all(train_set)
    x_valid, y_valid = load_all(valid_set)
    x_test, y_test = load_all(test_set)
    clf = svm.SVC(kernel='linear', gamma='scale') # kernel = 'linear', 'poly', 'rbf'. gamma = 'scale', 'auto'
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_train)
    print("Train accuracy:",metrics.accuracy_score(y_train, y_pred))

    y_pred = clf.predict(x_test)
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    # clf.predict(np.expand_dims(x_test[0, :], axis=0))

    dump(clf, path_save_model.format(name.split("_")[0], name))
    clf_load = load(path_save_model.format(name.split("_")[0], name))
    y_pred = clf.predict(x_test)
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))