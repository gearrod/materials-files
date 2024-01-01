from sklearn.decomposition import PCA, FastICA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from materials import TRAIN_DATA
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input, LeakyReLU
from keras.optimizers import Adam, Adadelta, SGD, Nadam
from sklearn_genetic.genetic_search import GASearchCV
from sklearn_genetic.space import Integer, Categorical, Continuous
from keras.losses import mean_squared_error
from scikeras.wrappers import KerasRegressor
from keras.callbacks import EarlyStopping
from keras.metrics import MeanSquaredError
import numpy as np

INVERSE_INPUT_DIM = 3
INVERSE_OUTPUT_DIM = 4

direct = Pipeline(
    (
        ("scaler", StandardScaler()),
        ("ICA", FastICA(n_components=4)),
        ("regressor", KNeighborsRegressor(weights="distance")),
    )
).fit(
    TRAIN_DATA[["angle", "ratio", "p_matrix", "p_fiber"]],
    TRAIN_DATA[["p11", "p22", "p12"]],
)


def inverse_nn(
    density: int,
    width: int,
    early_dropout: float,
    late_dropout: float,
    leaky_start: bool,
    leaky_end: bool,
):
    model = Sequential()

    model.add(Input(INVERSE_INPUT_DIM))

    if leaky_start:
        model.add(LeakyReLU())
    model.add(Dense(width, activation="relu"))
    model.add(Dropout(early_dropout))

    for i in range(width, INVERSE_OUTPUT_DIM, -(width // density)):
        model.add(Dense(i, activation="relu"))

    model.add(Dropout(late_dropout))
    if leaky_end:
        model.add(LeakyReLU())
    model.add(Dense(INVERSE_OUTPUT_DIM, activation="linear"))

    return model


INVERSE_PIPE = Pipeline(
    [
        ("scaling", StandardScaler()),
        ("regressor", KerasRegressor()),
    ]
)

INVERSE_PARAM_GRID = {
    "regressor": Categorical(
        [
            KerasRegressor(
                inverse_nn,
                epochs=40,
                batch_size=10000,
                metrics=MeanSquaredError,
                callbacks=EarlyStopping("mean_squared_error"),
            )
        ]
    ),
    "regressor__model__density": Integer(1, 10),
    "regressor__model__width": Integer(10, 1000),
    "regressor__model__early_dropout": Continuous(0.0, 0.8),
    "regressor__model__late_dropout": Continuous(0.0, 0.8),
    "regressor__model__leaky_start": Categorical([True, False]),
    "regressor__model__leaky_end": Categorical([True, False]),
    "regressor__optimizer": Categorical([Adam]),
    "regressor__optimizer__learning_rate": Continuous(0.0001, 0.1),
    "regressor__loss": Categorical([mean_squared_error]),
}

INVERSE_GRID = GASearchCV(
    INVERSE_PIPE,
    param_grid=INVERSE_PARAM_GRID,
    scoring="neg_root_mean_squared_error",
    cv=3,
    verbose=3,
    n_jobs=-1,
    population_size=10,
    return_train_score=True,
)

final = Pipeline([("scaler", StandardScaler()), ("preprocessing")])
