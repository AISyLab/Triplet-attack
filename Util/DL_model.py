from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *

def pick(classifier, input_shape, embedding_size):
    if classifier == 'Triplet':
        return Triplet(input_shape, embedding_size)
    else:
        print('No classifer avaliable')

def Triplet(input_shape, embedding_size):
    model = Sequential()
    model.add(Conv1D(input_shape=input_shape,filters=64,kernel_size=15,padding="same", activation="selu"))
    model.add(AveragePooling1D(pool_size=15,strides=15))
    model.add(Conv1D(filters=128, kernel_size=3, padding="same", activation="selu"))
    model.add(AveragePooling1D(pool_size=2,strides=2))
    model.add(Flatten(name='Flatten'))
    model.add(Dense(embedding_size))
    return model
