from dataclasses import dataclass
from sklearn.base import OneToOneFeatureMixin, RegressorMixin, TransformerMixin
from typing import Any, Dict, Type

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(init=False)
class PipelineConfig:
    scaler: OneToOneFeatureMixin
    preprocessing: TransformerMixin
    regressor: RegressorMixin
    scaler_name: str
    preprocessing_name: str
    regressor_name: str

    def __init__(self, scaler, preprocessing, regressor) -> None:
        self.scaler_name, self.scaler = scaler
        self.preprocessing_name, self.preprocessing = preprocessing
        self.regressor_name, self.regressor = regressor


@dataclass
class PipelineFactory:
    config: PipelineConfig

    def build(self):
        return Pipeline(
            (
                ("scaler", self.config.scaler),
                ("preprocessing", self.config.preprocessing),
                ("regressor", self.config.regressor),
            )
        )
