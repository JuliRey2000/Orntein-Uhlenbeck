"""Estados del par (4.5) de extremo a extremo con ``analyze_pair``: estado, causas y descartes."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from conftest import cointegrated_log_prices
from integration_helpers import (
    broken_relationship,
    explosive_spread,
    random_walk_and_stationary,
    series_pair,
    synthetic_preset,
)

from mrlab.analysis.pair_pipeline import (
    ILLUSTRATIVE_BANNER,
    STATUS_LABELS,
    STATUS_PRECEDENCE,
    AnalysisConfig,
    Cause,
    PairStatus,
    analyze_pair,
    resolve_status,
)
from mrlab.backtest.split import compute_split
from mrlab.domain.models import Segment
from mrlab.model.ou import OUStatus
from mrlab.stats.cointegration import Direction

PROMOTIONAL = ("garantizada", "par rentable", "estrategia ganadora", "valor justo")


def codes(causes: list[Cause]) -> list[str]:
    return [c.code for c in causes]


def all_text(analysis) -> str:
    return " ".join(
        [c.message for c in analysis.causes + analysis.discard_reasons] + analysis.warnings
    ).lower()


# --------------------------------------------------------------------------- ✅


def test_cointegrated_synthetic_pair_has_evidence_and_allows_strategy():
    a, b = synthetic_preset("SINT-COI")
    an = analyze_pair(a, b)

    assert an.status == PairStatus.EVIDENCE
    assert codes(an.causes) == ["EVIDENCIA_COINTEGRACION"]
    assert an.discard_reasons == []
    assert an.strategy_allowed and not an.illustrative_only
    assert ILLUSTRATIVE_BANNER not in an.warnings
    # Supuestos y modelo
    assert {r.classification for r in an.integration.values()} == {"compatible_I1"}
    assert an.eg_p_adjusted[an.chosen_direction] <= 0.05
    assert an.hedge is not None and abs(an.hedge.beta - 1.0) < 0.1
    assert an.ou is not None and an.ou.valid and 3 < an.ou.half_life < 15
    # z con μ y σ_eq FIJOS sobre toda la muestra; pronósticos a 1, 5 y 20
    assert an.zscore is not None and an.zscore.index.equals(an.spread.index)
    assert [f.horizon for f in an.forecasts] == [1, 5, 20]
    assert an.residuals is not None and an.ou_stability is not None
    # Datos sintéticos: el aviso viaja con el análisis
    assert any("SINTÉTICO" in w for w in an.warnings)
    assert not any(word in all_text(an) for word in ("garantizada", "par rentable"))


def test_evidence_uses_only_the_training_sample():
    a, b = synthetic_preset("SINT-COI")
    an = analyze_pair(a, b)
    split = an.split
    assert split.n_train == 900 and split.n_validation == 300 and split.n_test == 300
    assert all(eg.nobs == split.n_train for eg in an.eg.values())
    assert an.hedge.nobs == split.n_train and an.hedge.estimation_end == split.train_end
    assert an.ou.nobs == split.n_train - 1
    # El spread de formación tiene media ≈ 0 (α ya está dentro: consistencia α–μ)
    train_spread = an.spread.loc[: str(split.train_end)]
    assert abs(train_spread.mean()) < 1e-10
    assert abs(an.ou.mu) < 3 * an.ou.mu_se


# --------------------------------------------------------------------------- ⚠️ evidencia insuficiente


def test_independent_random_walks_have_insufficient_evidence_with_cause():
    a, b = synthetic_preset("SINT-IND")
    an = analyze_pair(a, b)

    assert {r.classification for r in an.integration.values()} == {"compatible_I1"}
    assert an.status == PairStatus.INSUFFICIENT_EVIDENCE
    first = an.causes[0]
    assert first.code == "EVIDENCIA_INSUFICIENTE"
    p_adj = an.eg_p_adjusted[an.chosen_direction]
    assert p_adj > 0.05
    assert f"{p_adj:.4f}".replace(".", ",") in first.message and "> 0,05" in first.message
    assert "Benjamini–Hochberg" in first.message and "m = 2" in first.message
    # Todas las causas activas se listan, en orden de precedencia
    assert codes(an.causes)[1:] and set(codes(an.causes)[1:]) <= {
        "VENTANAS_FALLIDAS",
        "BETA_CAMBIA_SIGNO",
    }
    assert codes(an.discard_reasons)[: len(an.causes)] == codes(an.causes)
    assert an.illustrative_only
    if an.strategy_allowed:  # OU válido → backtest permitido, pero solo ilustrativo (O5)
        assert ILLUSTRATIVE_BANNER in an.warnings


# --------------------------------------------------------------------------- ⛔ datos


def test_different_currencies_block_with_currency_cause():
    log_a, log_b = cointegrated_log_prices(n=1000, seed=1)
    a, b = series_pair(log_a, log_b, currency_b="EUR")
    an = analyze_pair(a, b)

    assert an.status == PairStatus.INSUFFICIENT_DATA
    assert codes(an.causes) == ["MONEDA_DISTINTA"]
    assert "USD" in an.causes[0].message and "EUR" in an.causes[0].message
    assert "extensión" in an.causes[0].message
    assert not an.strategy_allowed and an.illustrative_only
    assert an.aligned is None and an.hedge is None and an.ou is None and an.eg == {}
    assert an.validation.currency_mismatch and an.validation.blocked


def test_unknown_currency_blocks():
    log_a, log_b = cointegrated_log_prices(n=1000, seed=1)
    a, b = series_pair(log_a, log_b, currency_b=None)
    an = analyze_pair(a, b)
    assert an.status == PairStatus.INSUFFICIENT_DATA
    assert codes(an.causes) == ["MONEDA_DESCONOCIDA"]


def test_short_training_sample_blocks_with_numbers():
    log_a, log_b = cointegrated_log_prices(n=700, seed=2)
    a, b = series_pair(log_a, log_b)
    an = analyze_pair(a, b)

    assert an.status == PairStatus.INSUFFICIENT_DATA
    assert codes(an.causes) == ["DATOS_INSUFICIENTES"]
    msg = an.causes[0].message
    for number in ("700", "420", "500", "834"):  # comunes, formación, mínimo, total necesario
        assert number in msg
    assert not an.strategy_allowed
    assert an.aligned is not None  # se puede graficar aunque no se analice
    assert an.eg == {} and an.ou is None


def test_non_positive_prices_block_log_spread():
    log_a, log_b = cointegrated_log_prices(n=1000, seed=3)
    a, b = series_pair(log_a, log_b)
    data = a.data.copy()
    data.iloc[10, 0] = 0.0
    an = analyze_pair(a.with_data(data), b)
    assert an.status == PairStatus.INSUFFICIENT_DATA
    assert codes(an.causes) == ["SPREAD_LOG_BLOQUEADO"]


def test_frozen_split_shorter_than_minimum_blocks():
    log_a, log_b = cointegrated_log_prices(n=1000, seed=4)
    a, b = series_pair(log_a, log_b)
    short = compute_split(a.data.index[:700])  # formación de 420 observaciones
    an = analyze_pair(a, b, frozen_split=short)
    assert an.status == PairStatus.INSUFFICIENT_DATA
    assert "420" in an.causes[0].message and "500" in an.causes[0].message
    assert an.split == short


# --------------------------------------------------------------------------- ⚠️ supuestos


def test_stationary_series_gives_doubtful_assumptions():
    log_a, log_b = random_walk_and_stationary(seed=0)
    a, b = series_pair(log_a, log_b)
    an = analyze_pair(a, b)

    assert an.integration["AAA"].classification == "compatible_I1"
    assert an.integration["BBB"].classification == "posiblemente_estacionaria"
    assert an.status == PairStatus.DOUBTFUL_ASSUMPTIONS
    first = an.causes[0]
    assert first.code == "SERIE_NO_I1" and "«BBB»" in first.message
    assert "ADF" in first.message and "KPSS" in first.message
    assert an.illustrative_only


# --------------------------------------------------------------------------- ⚠️ inestable


def test_structural_break_gives_unstable_result():
    an = analyze_pair(*series_pair(*broken_relationship(seed=10)))
    assert an.status == PairStatus.UNSTABLE
    assert codes(an.causes) == ["VENTANAS_FALLIDAS"]
    assert an.eg_p_adjusted[an.chosen_direction] <= 0.05  # en la muestra completa sí hay evidencia
    assert an.stability.frac_fail > 0.40
    assert "ventanas" in an.causes[0].message and "40 %" in an.causes[0].message
    assert an.illustrative_only

    # El mismo par sin la ruptura es estable y tiene evidencia: la causa es la ruptura.
    control = analyze_pair(*series_pair(*broken_relationship(seed=10, break_end=300)))
    assert control.status == PairStatus.EVIDENCE
    assert control.stability.frac_fail <= 0.40


# --------------------------------------------------------------------------- OU no válido


def test_non_stationary_spread_blocks_strategy_without_half_life_or_bands():
    log_a, log_b = explosive_spread(seed=1)
    a, b = series_pair(log_a, log_b)
    an = analyze_pair(a, b, AnalysisConfig(direction=Direction.A_ON_B))

    assert an.ou is not None and an.ou.status == OUStatus.NON_STATIONARY
    assert an.ou.b >= 1 and not an.ou.valid
    assert not an.strategy_allowed
    reason = next(c for c in an.discard_reasons if c.code == "OU_NO_ESTACIONARIO")
    assert reason.status is None
    assert "≥ 1" in reason.message and "no muestra reversión estacionaria" in reason.message
    # Sin vida media, λ, σ_eq, μ, z, bandas ni pronósticos
    ou = an.ou
    assert ou.half_life is None and ou.lam is None and ou.sigma_eq is None and ou.mu is None
    assert an.zscore is None and an.forecasts == [] and an.residuals is None
    assert an.ou_stability is None
    # El estado sigue siendo uno de los cinco (el OU no es un estado)
    assert an.status in STATUS_LABELS


# --------------------------------------------------------------------------- precedencia


def test_precedence_insufficient_data_over_currency_lists_every_cause():
    log_a, log_b = cointegrated_log_prices(n=600, seed=5)
    a, b = series_pair(log_a, log_b, currency_b="EUR")
    an = analyze_pair(a, b)
    assert an.status == PairStatus.INSUFFICIENT_DATA
    assert set(codes(an.causes)) == {"MONEDA_DISTINTA", "DATOS_INSUFICIENTES"}


def test_precedence_doubtful_over_unstable_lists_every_cause():
    an = analyze_pair(*series_pair(*random_walk_and_stationary(seed=0)))
    assert an.status == PairStatus.DOUBTFUL_ASSUMPTIONS
    statuses = [c.status for c in an.causes]
    assert PairStatus.UNSTABLE in statuses  # la causa de menor precedencia también se lista
    order = [STATUS_PRECEDENCE.index(s) for s in statuses]
    assert order == sorted(order)


def test_precedence_insufficient_evidence_over_unstable():
    an = analyze_pair(*synthetic_preset("SINT-IND"))
    statuses = {c.status for c in an.causes}
    assert {PairStatus.INSUFFICIENT_EVIDENCE, PairStatus.UNSTABLE} <= statuses
    assert an.status == PairStatus.INSUFFICIENT_EVIDENCE


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_resolve_status_follows_precedence_for_any_combination(k):
    blocking = STATUS_PRECEDENCE[:-1]
    for combo in itertools.combinations(blocking, k):
        for perm in itertools.permutations(combo):
            causes = [Cause(code=s.value, status=s, message="x") for s in perm]
            expected = min(combo, key=STATUS_PRECEDENCE.index)
            assert resolve_status(causes) == expected
    assert resolve_status([]) == PairStatus.EVIDENCE
    assert resolve_status([Cause(code="INFO", status=None, message="x")]) == PairStatus.EVIDENCE


# --------------------------------------------------------------------------- otros descartes


def test_correlation_filter_is_only_a_discard_reason():
    a, b = synthetic_preset("SINT-CBC")  # cointegrado con baja correlación
    base = analyze_pair(a, b)
    filtered = analyze_pair(a, b, AnalysisConfig(correlation_filter=True, correlation_min=0.5))
    assert filtered.correlation_train is not None and filtered.correlation_train < 0.5
    assert filtered.status == base.status  # no cambia el estado
    reason = next(c for c in filtered.discard_reasons if c.code == "FILTRO_CORRELACION")
    assert reason.status is None and "no es una prueba de cointegración" in reason.message
    assert "FILTRO_CORRELACION" not in codes(base.discard_reasons)


def test_fixed_direction_is_respected_and_reported():
    a, b = synthetic_preset("SINT-COI")
    an = analyze_pair(a, b, AnalysisConfig(direction=Direction.B_ON_A))
    assert an.chosen_direction == Direction.B_ON_A and not an.direction_was_auto
    assert an.hedge.dependent == "SINT-COI-B"
    assert "fijada por el usuario" in an.eg_decision.message


def test_executable_proxy_is_aligned_and_warned():
    a, b = synthetic_preset("SINT-COI")
    proxy = a.with_meta(asset=a.meta.asset.model_copy(update={"symbol": "PROXY-A"}))
    proxy = proxy.with_data(a.data.drop(a.data.index[[5, 6]]) * 1.01)
    an = analyze_pair(a, b, exec_a=proxy)
    assert an.exec_symbol_a == "PROXY-A" and an.exec_symbol_b is None
    assert len(an.exec_prices) == len(a.data) - 2
    assert an.exec_prices.index.equals(an.aligned.prices.index)
    assert np.allclose(an.exec_prices["a"], an.aligned.prices["a"] * 1.01)
    assert any("riesgo de base" in w for w in an.warnings)
    assert any("no tienen precio del instrumento ejecutable" in w for w in an.warnings)


def test_no_promotional_language_in_any_message():
    for prefix in ("SINT-COI", "SINT-IND", "SINT-CNC", "SINT-CBC"):
        an = analyze_pair(*synthetic_preset(prefix))
        text = all_text(an)
        for phrase in PROMOTIONAL:
            # «valor justo» solo puede aparecer negado («no un valor justo»)
            assert phrase not in text.replace("no un valor justo", "")


def test_segments_cover_the_whole_sample():
    a, b = synthetic_preset("SINT-COI")
    an = analyze_pair(a, b)
    total = sum(
        len(an.spread.loc[str(s) : str(e)])
        for s, e in (
            an.split.bounds(seg) for seg in (Segment.TRAIN, Segment.VALIDATION, Segment.TEST)
        )
    )
    assert total == len(an.spread) == 1500
