from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import make_scorer, r2_score, mean_squared_error

SCORING = {
    "R2": make_scorer(r2_score),
    "MSE": make_scorer(mean_squared_error, greater_is_better=False),
}

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

PIPE_NAMES_MAPPING = {
    PCA: "PCA",
    FastICA: "ICA",
    FunctionTransformer: "Ninguno",
    LinearRegression: "Regresión lineal",
    DecisionTreeRegressor: "Árbol de decisión",
    KNeighborsRegressor: "KNN",
    RandomForestRegressor: "Random Forest",
}


def get_pipe_name(pipe):
    return (
        (
            PIPE_NAMES_MAPPING[pipe["preprocessing"].__class__]
            + "_"
            + f"_".join(
                str(v) for k, v in pipe.items() if k.startswith("preprocessing__")
            )
            + " + "
            + PIPE_NAMES_MAPPING[pipe["regressor"].__class__]
            + "_"
            + f"_".join(str(v) for k, v in pipe.items() if k.startswith("regressor__"))
        )
        if pipe["preprocessing"].__class__ != FunctionTransformer
        else (
            PIPE_NAMES_MAPPING[pipe["regressor"].__class__]
            + "_"
            + f"_".join(str(v) for k, v in pipe.items() if k.startswith("regressor__"))
        )
    )


INVERSE_GRID = GridSearchCV(
    INVERSE_PIPE,
    n_jobs=-1,
    param_grid=INVERSE_PARAM_GRID,
    scoring=SCORING,
    refit="R2",
    verbose=4,
    return_train_score=True,
)

INVERSE_GRID_SILENT = GridSearchCV(
    INVERSE_PIPE,
    n_jobs=-1,
    param_grid=INVERSE_PARAM_GRID,
    scoring=SCORING,
    refit="R2",
    verbose=0,
    return_train_score=True,
)
