import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.model_selection import KFold, ParameterGrid

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import make_scorer, r2_score, mean_squared_error
from certain import certain, direct_model_trained
import numpy as np

INVERSE_PIPE = Pipeline(
    [
        ("scaling", StandardScaler()),
        # the preprocessing stage is populated by the param_grid
        ("preprocessing", "passthrough"),
        # idem for the regressor
        ("regressor", "passthrough"),
    ]
)

N_FEATURES_OPTIONS = [2, 3]

INVERSE_PARAM_GRID = [
    {
        "preprocessing": [PCA(), FastICA()],
        "preprocessing__n_components": N_FEATURES_OPTIONS,
        "regressor": [
            LinearRegression(),
            DecisionTreeRegressor(),
        ],
    },
    {
        "preprocessing": [FunctionTransformer(lambda x: x)],
        "regressor": [
            LinearRegression(),
            DecisionTreeRegressor(),
        ],
    },
    {
        "preprocessing": [PCA(), FastICA()],
        "preprocessing__n_components": N_FEATURES_OPTIONS,
        "regressor": [
            KNeighborsRegressor(),
        ],
        "regressor__n_neighbors": [5, 6, 7, 8, 9, 10, 11, 12, 13],
        "regressor__weights": ["uniform", "distance"],
    },
    {
        "preprocessing": [FunctionTransformer(lambda x: x)],
        "regressor": [
            KNeighborsRegressor(),
        ],
        "regressor__n_neighbors": [1, 4, 5, 6, 9],
        "regressor__weights": ["uniform", "distance"],
    },
    {
        "preprocessing": [PCA(), FastICA()],
        "preprocessing__n_components": N_FEATURES_OPTIONS,
        "regressor": [
            RandomForestRegressor(),
        ],
        "regressor__n_estimators": [5, 20, 50],
    },
    {
        "preprocessing": [FunctionTransformer(lambda x: x)],
        "regressor": [
            RandomForestRegressor(),
        ],
        "regressor__n_estimators": [5, 20, 50],
    },
]


def run_with_params(params, x, y):
    kFold = KFold(5)

    x = x.to_numpy()
    y = y.to_numpy()

    results = []

    for i, (train, test) in enumerate(kFold.split(x, y)):
        print(f"Split{i}: ", end="")

        start_time = time.time()

        split_x, split_y = (x[train, :], y[train, :])
        pipe = INVERSE_PIPE.set_params(**params)
        pipe.fit(split_x, split_y)

        train_time = time.time() - start_time

        print(f"Train done in {train_time:.4f} seconds.\t", end="")

        split_x_test = x[test, :]

        start_time = time.time()

        pred_test = pipe.predict(split_x_test)

        pred_time = time.time() - start_time

        print(f"Prediction done in {pred_time:.4f} seconds.\t", end="")

        certain_score = certain(split_x_test, pred_test)

        r = {
            "x_train": split_x,
            "y_train": split_y,
            "x_test": split_x_test,
            "y_test": y[test, :],
            "predicted_test": pred_test,
            "time_train": train_time,
            "time_pred": pred_time,
            "certain": {
                "values": certain_score,
                "min": np.min(certain_score),
                "max": np.max(certain_score),
                "mean": np.mean(certain_score),
                "median": np.median(certain_score),
                "variance": np.var(certain_score),
            },
        }

        print(f"Mean certain {r['certain']['mean']:.3f}.\t")

        results.append(r)

    return {
        "splits": results,
        "mean_certain": np.concatenate(
            [r["certain"]["values"] for r in results]
        ).mean(),
    }


def exhaustive_search(x, y):
    results = []

    for params in ParameterGrid(INVERSE_PARAM_GRID):
        print(f"Params:{str(params)}")
        r = run_with_params(params, x, y)
        print("")
        results.append({"params": params, "results": r})

    best_params = max((r for r in results), key=lambda r: r["results"]["mean_certain"])
    best_score = best_params["results"]["mean_certain"]
    best_params = best_params["params"]

    return (results, best_params, best_score)
