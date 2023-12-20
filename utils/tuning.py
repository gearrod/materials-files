# Code related to hyper parameter tuning of the KFold wrapper
from typing import Any, Dict
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ExpSineSquared, Matern
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import (
    PolynomialFeatures,
    StandardScaler,
    FunctionTransformer,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from utils.kfold import KFoldLoop
from utils.pipeline import PipelineConfig, PipelineFactory
from utils.nn import build_neural_n
import numpy as np

from sklearn.model_selection import ParameterGrid
from sklearn.neural_network import MLPRegressor

# Define the experiment grid
GRID = [
    {
        "scaler": [("scaler", StandardScaler())],
        "preprocessing": [
            ("Original Features", FunctionTransformer(lambda x: x)),
            ("Degree 2 Polynomial", PolynomialFeatures(degree=2)),
            (
                "Logarithmic+Cos",
                FunctionTransformer(
                    lambda x: np.concatenate([np.log(np.abs(x)), np.cos(x), x], axis=1)
                ),
            ),
        ],
        "regressor": [
            ("Baseline", LinearRegression()),
            ("DT", DecisionTreeRegressor()),
            ("HBGB LR=0.1", HistGradientBoostingRegressor(learning_rate=0.1)),
            ("HBGB LR=0.4", HistGradientBoostingRegressor(learning_rate=0.4)),
            ("HBGB LR=0.05", HistGradientBoostingRegressor(learning_rate=0.05)),
            ("RF", RandomForestRegressor()),
            ("RF 250", RandomForestRegressor(n_estimators=250)),
            ("RF MX40", RandomForestRegressor(n_estimators=500, max_depth=40)),
            # (""),
        ],
    }
]

GRID_ITERATIONS = list(ParameterGrid(GRID))


# Create a KFoldLoop instance using specific parameters from the grid
def build_iteration(hyper_parameters: Dict[str, Any]):
    pipeline_config = PipelineConfig(**hyper_parameters)
    pipeline_factory = PipelineFactory(config=pipeline_config)
    return KFoldLoop(pipeline_factory=pipeline_factory)


TUNING_ITERATIONS = (
    build_iteration(hyper_parameters) for hyper_parameters in GRID_ITERATIONS
)
