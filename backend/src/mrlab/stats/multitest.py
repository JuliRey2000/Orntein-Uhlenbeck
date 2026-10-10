"""Corrección por pruebas múltiples (4.3).

CONTRATO (a implementar SIN usar statsmodels; las pruebas lo comparan con
``statsmodels.stats.multitest.multipletests``):
- ``benjamini_hochberg(pvalues, q)`` → (rechaza: ndarray[bool], p_ajustado: ndarray[float])
  p_ajustado_(i) = min_{k≥i} ( m·p_(k)/k ), acotado a 1, en el orden original.
- ``benjamini_yekutieli``: igual con factor c(m) = Σ 1/k (dependencia arbitraria).
- Acepta listas o arrays; vacío → arrays vacíos; NaN → ValueError en español.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np

FDR_EXPLANATION = (
    "Si pruebas muchos pares, algunos parecerán cointegrados por puro azar: con 100 pares "
    "independientes y un umbral del 5 %, esperarías unos 5 falsos positivos. La corrección de "
    "Benjamini–Hochberg ajusta los p-values para controlar la proporción esperada de falsos "
    "descubrimientos (FDR) entre los pares que declaras significativos."
)


def benjamini_hochberg(pvalues: Sequence[float] | np.ndarray, q: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError


def benjamini_yekutieli(pvalues: Sequence[float] | np.ndarray, q: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError


def adjust_pvalues(
    pvalues: Sequence[float] | np.ndarray, method: Literal["bh", "by"] = "bh", q: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    raise NotImplementedError
