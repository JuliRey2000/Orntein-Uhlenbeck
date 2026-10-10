"""Corrección por pruebas múltiples (4.3).

CONTRATO (a implementar SIN usar statsmodels; las pruebas lo comparan con
``statsmodels.stats.multitest.multipletests``):
- ``benjamini_hochberg(pvalues, q)`` → (rechaza: ndarray[bool], p_ajustado: ndarray[float])
  p_ajustado_(i) = min_{k≥i} ( m·p_(k)/k ), acotado a 1, en el orden original.
- ``benjamini_yekutieli``: igual con factor c(m) = Σ 1/k (dependencia arbitraria).
- Acepta listas o arrays; vacío → arrays vacíos; NaN → ValueError en español.

Notas de implementación:
- ``rechaza`` se obtiene con el procedimiento escalonado clásico (se rechazan
  las k primeras pruebas ordenadas, donde k es la mayor con
  p_(k) ≤ q·k/(m·c(m))). Matemáticamente equivale a ``p_ajustado ≤ q``; se usa
  la forma escalonada para coincidir bit a bit con statsmodels.
- Los p-values ajustados no dependen de ``q``: ``q`` solo decide qué se rechaza.
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

FDR_METHOD_LABELS = {
    "bh": "Benjamini–Hochberg",
    "by": "Benjamini–Yekutieli",
}


def _validate(pvalues: Sequence[float] | np.ndarray, q: float) -> np.ndarray:
    if not 0.0 < q < 1.0:
        raise ValueError(
            f"La tasa de falsos descubrimientos q debe estar entre 0 y 1 (se recibió {q})."
        )
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1:
        raise ValueError("Los p-values deben ser una lista o un vector de una dimensión.")
    if np.isnan(p).any():
        raise ValueError(
            "Hay p-values faltantes (NaN): no se puede aplicar la corrección por pruebas múltiples. "
            "Revisa qué prueba no pudo calcularse antes de ajustar."
        )
    if ((p < 0.0) | (p > 1.0)).any():
        raise ValueError("Todos los p-values deben estar entre 0 y 1.")
    return p


def _step_up(p: np.ndarray, q: float, dependence_factor: float) -> tuple[np.ndarray, np.ndarray]:
    """Procedimiento escalonado de BH con p ordenados internamente; devuelve en el orden original."""
    m = p.size
    if m == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=float)

    order = np.argsort(p)
    p_sorted = p[order]
    # Umbral relativo de la k-ésima prueba ordenada: k / (m · c(m)).
    ecdf_factor = np.arange(1, m + 1) / m / dependence_factor

    reject_sorted = p_sorted <= ecdf_factor * q
    if reject_sorted.any():
        reject_sorted[: np.nonzero(reject_sorted)[0].max()] = True

    adjusted_sorted = np.minimum.accumulate((p_sorted / ecdf_factor)[::-1])[::-1]
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)

    reject = np.empty(m, dtype=bool)
    adjusted = np.empty(m, dtype=float)
    reject[order] = reject_sorted
    adjusted[order] = adjusted_sorted
    return reject, adjusted


def benjamini_hochberg(
    pvalues: Sequence[float] | np.ndarray, q: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini–Hochberg: controla la FDR bajo independencia o dependencia positiva."""
    p = _validate(pvalues, q)
    return _step_up(p, q, dependence_factor=1.0)


def benjamini_yekutieli(
    pvalues: Sequence[float] | np.ndarray, q: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini–Yekutieli: válido bajo dependencia arbitraria (factor c(m) = Σ_{k=1..m} 1/k)."""
    p = _validate(pvalues, q)
    c_m = float(np.sum(1.0 / np.arange(1, p.size + 1))) if p.size else 1.0
    return _step_up(p, q, dependence_factor=c_m)


def adjust_pvalues(
    pvalues: Sequence[float] | np.ndarray, method: Literal["bh", "by"] = "bh", q: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    """Despacha a BH (``"bh"``) o BY (``"by"``). Devuelve (rechaza, p_ajustado)."""
    if method == "bh":
        return benjamini_hochberg(pvalues, q)
    if method == "by":
        return benjamini_yekutieli(pvalues, q)
    raise ValueError(
        f"Método de corrección desconocido: {method!r}. Usa 'bh' (Benjamini–Hochberg) "
        "o 'by' (Benjamini–Yekutieli)."
    )
