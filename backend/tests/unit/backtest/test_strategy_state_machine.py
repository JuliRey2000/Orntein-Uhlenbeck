"""Funciones puras de la máquina de estados (7.2, 7.5, S1, S2)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from mrlab.strategy.params import StrategyParams
from mrlab.strategy.state_machine import (
    EXIT_PRIORITY,
    EXIT_REASON_LABELS,
    REARM_REASONS,
    STOP_REASONS,
    ExitReason,
    Side,
    active_exit_conditions,
    entry_signal,
    rearm_cleared,
    select_exit,
)

P = StrategyParams()  # entrada 2, salida 0,5, stop 3,5, stop monetario 5 % de 10.000
MAX = 30


def exits(side, z, pnl=0.0, bars=0):
    return active_exit_conditions(side, z, pnl, bars, MAX, P)


@pytest.mark.parametrize(
    ("z", "expected"),
    [
        (2.01, Side.SHORT),
        (-2.01, Side.LONG),
        (2.0, None),
        (-2.0, None),
        (0.0, None),
        (math.nan, None),
    ],
)
def test_entry_signal(z, expected):
    assert entry_signal(z, P) == expected


def test_short_exits_are_directional():
    assert exits(Side.SHORT, 3.5) == [ExitReason.STOP_Z]
    assert exits(Side.SHORT, 0.5) == [ExitReason.NORMAL]
    # S1: z salta de +2,5 a −1 → el corto sale (la lectura literal |z| ≤ 0,5 lo dejaría abierto)
    assert exits(Side.SHORT, -1.0) == [ExitReason.NORMAL]
    assert exits(Side.SHORT, -5.0) == [ExitReason.NORMAL]
    assert exits(Side.SHORT, 1.0) == []
    assert exits(Side.SHORT, 3.49) == []


def test_long_exits_are_directional():
    assert exits(Side.LONG, -3.5) == [ExitReason.STOP_Z]
    assert exits(Side.LONG, -0.5) == [ExitReason.NORMAL]
    assert exits(Side.LONG, 1.0) == [ExitReason.NORMAL]
    assert exits(Side.LONG, 5.0) == [ExitReason.NORMAL]
    assert exits(Side.LONG, -1.0) == []
    assert exits(Side.LONG, -3.49) == []


def test_money_stop_and_max_time_thresholds():
    limit = -P.stop_loss_pct * P.initial_capital  # −500
    assert exits(Side.SHORT, 1.0, pnl=limit) == [ExitReason.STOP_MONEY]
    assert exits(Side.SHORT, 1.0, pnl=limit + 0.01) == []
    assert exits(Side.SHORT, 1.0, bars=MAX) == [ExitReason.MAX_TIME]
    assert exits(Side.SHORT, 1.0, bars=MAX - 1) == []


def test_all_active_conditions_are_listed_in_priority_order():
    assert exits(Side.SHORT, 3.6, pnl=-600, bars=MAX) == [
        ExitReason.STOP_MONEY,
        ExitReason.STOP_Z,
        ExitReason.MAX_TIME,
    ]
    assert exits(Side.LONG, 0.2, pnl=-600, bars=MAX + 4) == [
        ExitReason.STOP_MONEY,
        ExitReason.MAX_TIME,
        ExitReason.NORMAL,
    ]


def test_z_stop_and_normal_exit_are_mutually_exclusive():
    """Con salida < entrada < stop y salidas direccionales nunca coinciden las 4 condiciones."""
    for side in Side:
        for z in np.linspace(-6, 6, 241):
            active = exits(side, float(z))
            assert not (ExitReason.STOP_Z in active and ExitReason.NORMAL in active)


def test_nan_z_only_checks_money_and_time():
    assert exits(Side.SHORT, math.nan, pnl=-600, bars=MAX) == [
        ExitReason.STOP_MONEY,
        ExitReason.MAX_TIME,
    ]
    assert exits(Side.LONG, math.nan) == []


@pytest.mark.parametrize(
    ("active", "expected"),
    [
        (
            [ExitReason.NORMAL, ExitReason.MAX_TIME, ExitReason.STOP_Z, ExitReason.STOP_MONEY],
            ExitReason.STOP_MONEY,
        ),
        ([ExitReason.NORMAL, ExitReason.MAX_TIME, ExitReason.STOP_Z], ExitReason.STOP_Z),
        ([ExitReason.NORMAL, ExitReason.MAX_TIME], ExitReason.MAX_TIME),
        ([ExitReason.NORMAL], ExitReason.NORMAL),
        ([ExitReason.NORMAL, ExitReason.FORCED, ExitReason.STOP_MONEY], ExitReason.FORCED),
        ([], None),
    ],
)
def test_select_exit_priority(active, expected):
    assert select_exit(active) == expected


def test_priority_constants():
    assert EXIT_PRIORITY[1:] == (
        ExitReason.STOP_MONEY,
        ExitReason.STOP_Z,
        ExitReason.MAX_TIME,
        ExitReason.NORMAL,
    )
    assert STOP_REASONS == {ExitReason.STOP_MONEY, ExitReason.STOP_Z}
    assert REARM_REASONS == {ExitReason.STOP_MONEY, ExitReason.STOP_Z, ExitReason.MAX_TIME}
    assert set(EXIT_REASON_LABELS) == set(ExitReason)


def test_rearm_cleared_per_side():
    assert rearm_cleared(Side.SHORT, 2.0, P)
    assert rearm_cleared(Side.SHORT, -3.0, P)
    assert not rearm_cleared(Side.SHORT, 2.01, P)
    assert rearm_cleared(Side.LONG, -2.0, P)
    assert rearm_cleared(Side.LONG, 3.0, P)
    assert not rearm_cleared(Side.LONG, -2.01, P)
    assert not rearm_cleared(Side.LONG, math.nan, P)
