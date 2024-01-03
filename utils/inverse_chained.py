from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import RegressorChain
from itertools import permutations
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import make_scorer, r2_score, mean_squared_error

SCORING = {
    "R2": make_scorer(r2_score),
    "MSE": make_scorer(mean_squared_error, greater_is_better=False),
}

CHAINED_INVERSE_PIPE = Pipeline(
    [
        # ("scaling", StandardScaler()),
        # the preprocessing stage is populated by the param_grid
        ("preprocessing", FastICA(n_components=3)),
        # idem for the regressor
        (
            "regressor",
            RegressorChain(base_estimator=KNeighborsRegressor(), order=None, cv=5),
        ),
    ]
)

CHAINED_INVERSE_PARAM_GRID = [
    {
        "regressor__base_estimator": [KNeighborsRegressor()],
        "regressor__base_estimator__n_neighbors": [
            1,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
        ],
        "regressor__order": [p for p in permutations(range(4), 4)],
    },
    {
        "regressor__base_estimator": [
            DecisionTreeRegressor(),
            # HistGradientBoostingRegressor(),
        ],
        "regressor__order": [p for p in permutations(range(4), 4)],
    },
]


def get_pipe_name(pipe):
    return (
        "->".join(
            (
                ["angle", "ratio", "p_matrix", "p_fiber"][i]
                for i in pipe["regressor"].order
            )
        )
        + f" {pipe['regressor'].base_estimator.n_neighbors}N"
    )


CHAINED_INVERSE_GRID = GridSearchCV(
    CHAINED_INVERSE_PIPE,
    n_jobs=-1,
    param_grid=CHAINED_INVERSE_PARAM_GRID,
    scoring=SCORING,
    refit="R2",
    verbose=4,
    return_train_score=True,
)

CHAINED_INVERSE_GRID_SILENT = GridSearchCV(
    CHAINED_INVERSE_PIPE,
    n_jobs=-1,
    param_grid=CHAINED_INVERSE_PARAM_GRID,
    scoring=SCORING,
    refit="R2",
    verbose=0,
    return_train_score=True,
)
