"""
Collection of classifier models.
Different architectures are trained
on face embeddings.
"""
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense,Dropout,Softmax,Flatten,Activation,BatchNormalization

# TODO: Add GMM and SVM classifier

def get_classifier(name, load_pretrained):
    """
    Returns classifier
    """
    AVAILABLE_CLASSIFIERS = {
    "tiny_2_layer" : get_tiny_2_layer,
    "2_layer_dense" : get_2_layer_dense
    }
    path_save_model = "./data/models/attack_classifier/{}.h5"
    if load_pretrained:
        return load_model(path_save_model.format(name))
    return AVAILABLE_CLASSIFIERS[name]()

def get_tiny_2_layer():
    model=Sequential(name="tiny_2_layer")
    model.add(Dense(units=4,input_dim=(128), kernel_initializer='glorot_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(units=1, kernel_initializer='he_uniform'))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model

def get_2_layer_dense():
    model=Sequential(name="2_layer_dense")
    model.add(Dense(units=64,input_dim=(128), kernel_initializer='glorot_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(units=1, kernel_initializer='he_uniform'))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model
