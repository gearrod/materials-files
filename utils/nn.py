from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras import activations
from tensorflow import math as tfm
import numpy as np
from keras.layers import LeakyReLU
from keras.losses import (
    mean_squared_error,
    binary_crossentropy,
    mean_absolute_percentage_error,
)


def build_neural_n(
    start=500,
    mid=125,
    end=50,
    dropout=0.2,
    standard_activation="relu",
    mid_activation="relu",
    mse=True,
):
    model = Sequential()

    model.add(Input(shape=(12,)))
    model.add(Dense(start, activation=standard_activation))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(mid, activation=mid_activation))
    model.add(Dense(end, activation=standard_activation))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation=standard_activation))

    model.compile(
        loss=mean_squared_error if mse else mean_absolute_percentage_error,
        optimizer="adam",
        metrics=["mse"],
    )

    return model
