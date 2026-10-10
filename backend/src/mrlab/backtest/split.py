"""División cronológica de la muestra (8.1): 60 % formación / 20 % validación / 20 % prueba.

Las fechas de corte se calculan una vez y se congelan en el RunConfig: si luego
se añaden datos, la formación no se mueve.
"""

from __future__ import annotations

import math
from datetime import date
from typing import TypeVar

import pandas as pd
from pydantic import BaseModel, Field, model_validator

from mrlab.domain.errors import ConfigError
from mrlab.domain.models import Segment

T = TypeVar("T", pd.Series, pd.DataFrame)


class SplitConfig(BaseModel):
    train: float = Field(0.6, gt=0, lt=1)
    validation: float = Field(0.2, ge=0, lt=1)
    test: float = Field(0.2, ge=0, lt=1)

    @model_validator(mode="after")
    def _sum_to_one(self) -> SplitConfig:
        if abs(self.train + self.validation + self.test - 1.0) > 1e-9:
            raise ConfigError(
                "Las proporciones de formación, validación y prueba deben sumar 1.",
                code="DIVISION_INVALIDA",
            )
        return self


class SampleSplit(BaseModel):
    train_start: date
    train_end: date
    validation_start: date | None
    validation_end: date | None
    test_start: date | None
    test_end: date | None
    n_train: int
    n_validation: int
    n_test: int

    def bounds(self, segment: Segment) -> tuple[date | None, date | None]:
        return {
            Segment.TRAIN: (self.train_start, self.train_end),
            Segment.VALIDATION: (self.validation_start, self.validation_end),
            Segment.TEST: (self.test_start, self.test_end),
        }[Segment(segment)]


def compute_split(index: pd.DatetimeIndex, cfg: SplitConfig | None = None) -> SampleSplit:
    """n_formación = floor(n·train), n_validación = floor(n·validation), prueba = resto."""
    cfg = cfg or SplitConfig()
    idx = pd.DatetimeIndex(index).sort_values()
    n = len(idx)
    n_train = int(math.floor(n * cfg.train))
    n_val = int(math.floor(n * cfg.validation)) if cfg.test > 0 else n - n_train
    n_test = n - n_train - n_val
    if n_train < 1:
        raise ConfigError(
            "No hay datos suficientes para la muestra de formación.", code="DATOS_INSUFICIENTES"
        )

    def _d(i: int) -> date:
        return idx[i].date()

    return SampleSplit(
        train_start=_d(0),
        train_end=_d(n_train - 1),
        validation_start=_d(n_train) if n_val > 0 else None,
        validation_end=_d(n_train + n_val - 1) if n_val > 0 else None,
        test_start=_d(n_train + n_val) if n_test > 0 else None,
        test_end=_d(n - 1) if n_test > 0 else None,
        n_train=n_train,
        n_validation=n_val,
        n_test=n_test,
    )


def segment_slice(obj: T, split: SampleSplit, segment: Segment | str) -> T:
    """Recorta una serie/DataFrame indexado por fecha al segmento (inclusive)."""
    start, end = split.bounds(Segment(segment))
    if start is None or end is None:
        return obj.iloc[0:0]
    return obj.loc[pd.Timestamp(start) : pd.Timestamp(end)]
