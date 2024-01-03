from materials import *
import numpy as np
from direct import DIRECT_GRID_SILENT
from functools import lru_cache
from sklearn.preprocessing import StandardScaler


@lru_cache()
def direct_model_trained():
    return DIRECT_GRID_SILENT.fit(
        GLOBAL_DATA[["angle", "ratio", "p_matrix", "p_fiber"]],
        GLOBAL_DATA[["p11", "p22", "p12"]],
    )


def certain(data, prediction):
    direct = direct_model_trained()

    true = data[["p11", "p22", "p12"]]
    pred = direct.predict(prediction)

    ss = StandardScaler().fit(true)
    true = ss.transform(true)
    pred = ss.transform(pred)

    certain = np.linalg.norm(true - pred, axis=1) + 1.0

    return certain
