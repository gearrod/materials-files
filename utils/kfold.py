# Wrapper of kfold using PipelineFactory and PipelineConfig
from dataclasses import dataclass
from sklearn.model_selection import cross_validate
from utils.pipeline import PipelineFactory
from utils.pipeline import PipelineConfig

from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.metrics import r2_score


@dataclass
class KFoldLoop:
    pipeline_factory: PipelineFactory
    folds: int = 5

    def run(self, X, y):
        pipeline = self.pipeline_factory.build()
        cv_results = cross_validate(
            pipeline,
            X,
            y,
            cv=self.folds,
            scoring={
                "MSE": make_scorer(mean_squared_error, greater_is_better=False),
                "R2": make_scorer(r2_score),
            },
            verbose=3,
            n_jobs=-1,
        )
        return cv_results
