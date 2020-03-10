"""
Data generator module for face
embeddings
"""
import numpy as np
import tensorflow as tf

class DataGenerator(tf.keras.utils.Sequence):
    """
    Custom data generator
    """
    def __init__(self, dataset, batch_size=8, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """
        Returns number of batches in the dataset
        """
        return int(np.floor(len(self.dataset) / self.batch_size))

    def __getitem__(self, index):
        """
        Returns one batch
        """
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        _tmp = [self.dataset[k] for k in indexes]
        return self.convert(_tmp)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.dataset))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def get_xy(self, record):
        return np.load(record[0]), record[1]
    
    def convert(self, records):
        x, y = [], []
        for record in records:
            _x, _y = self.get_xy(record)
            x.append(_x)
            y.append(_y)
        return np.array(x), np.array(y)

def load_all(records):
    """
    Loads all records to memory.
    Alternative to data generator class.
    """
    x, y = [], []
    for record in records:
        _x, _y = np.load(record[0]), record[1]
        x.append(_x)
        y.append(_y)
    return np.array(x), np.array(y)

if __name__ == "__main__":
    from data_process import pickle_read
    train_set = pickle_read("./data/print_attack/processed/train.pkl")
    valid_set = pickle_read("./data/print_attack/processed/valid.pkl")
    test_set = pickle_read("./data/print_attack/processed/test.pkl")
    # data generator usage
    gen_train = DataGenerator(dataset=train_set, batch_size=128, shuffle=True)
    gen_valid = DataGenerator(dataset=valid_set, batch_size=128, shuffle=True)
    gen_test = DataGenerator(dataset=test_set, batch_size=128)
    # data generator alternate
    x_train, y_train = load_all(train_set)
    x_valid, y_valid = load_all(valid_set)
    x_test, y_test = load_all(test_set)
