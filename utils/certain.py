from sklearn.decomposition import FastICA
from sklearn.neighbors import KNeighborsRegressor
from materials import *
import numpy as np
from direct import DIRECT_GRID_SILENT
from functools import lru_cache
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


@lru_cache()
def direct_model_trained():
    direct_pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("preprocessing", FastICA(4)),
            ("regressor", KNeighborsRegressor(4, weights="distance")),
        ]
    )

    direct_pipe.fit(
        GLOBAL_DATA[["angle", "ratio", "p_matrix", "p_fiber"]].to_numpy(),
        GLOBAL_DATA[["p11", "p22", "p12"]].to_numpy(),
    )

    return direct_pipe


def certain(x, prediction):
    direct = direct_model_trained()

    true = x
    pred = direct.predict(prediction)

    ss = StandardScaler().fit(true)
    true = ss.transform(true)
    pred = ss.transform(pred)

    certain = (np.linalg.norm(true - pred, axis=1) + 1.0) ** (-1)

    return certain
