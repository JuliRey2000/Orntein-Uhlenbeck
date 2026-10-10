"""Ventanas móviles compartidas por estabilidad de cointegración (4.4) y del OU (6.2)."""

from __future__ import annotations


def rolling_windows(n: int, window: int = 252, step: int = 63) -> list[tuple[int, int]]:
    """Ventanas semiabiertas [inicio, fin) de tamaño ``window`` cada ``step`` posiciones.

    Empiezan en 0; la última es la que aún cabe completa. Si n < window → [].
    """
    if window <= 0 or step <= 0:
        raise ValueError("window y step deben ser positivos")
    return [(s, s + window) for s in range(0, n - window + 1, step)]
