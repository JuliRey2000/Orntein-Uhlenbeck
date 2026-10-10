"""Estados del par (4.5) de extremo a extremo con ``analyze_pair``: estado, causas y descartes."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest
from conftest import cointegrated_log_prices, make_series
from integration_helpers import (
    StubData,
    as_non_executable,
    broken_relationship,
    explosive_spread,
    random_walk_and_stationary,
    series_pair,
    stub_config,
    synthetic_preset,
    with_symbol,
)
from statsmodels.tsa.adfvalues import mackinnoncrit

from mrlab.analysis.pair_pipeline import (
    ILLUSTRATIVE_BANNER,
    STATUS_LABELS,
    STATUS_PRECEDENCE,
    STRATEGY_BLOCKING_CODES,
    AnalysisConfig,
    Cause,
    PairStatus,
    analyze_pair,
    resolve_status,
)
from mrlab.analysis.runner import run_pair
from mrlab.backtest.split import compute_split
from mrlab.domain.errors import ConfigError
from mrlab.domain.models import AssetClass, CalendarType, Segment, SpreadType
from mrlab.model.ou import OUStatus, fit_ou
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
    assert an.illustrative_reasons == [] and an.illustrative_banner is None
    assert not any(w.startswith(ILLUSTRATIVE_BANNER) for w in an.warnings)
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
    # Decisión O7: el residuo de dos paseos independientes no se distingue de un paseo
    # aleatorio (valores críticos N = 2) → OU no válido, sin vida media y sin estrategia.
    assert an.ou.status == OUStatus.NON_STATIONARY and an.ou.near_unit_root
    assert "OU_NO_ESTACIONARIO" in codes(an.discard_reasons)
    assert an.ou.half_life is None and an.zscore is None
    assert not an.strategy_allowed
    assert not any(w.startswith(ILLUSTRATIVE_BANNER) for w in an.warnings)


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


# --------------------------------------------------------------------------- decisión O7


def _random_walk_pair(seed: int, n: int = 1500):
    rng = np.random.default_rng(seed)
    log_a = np.log(100) + np.cumsum(rng.normal(0, 0.01, n))
    log_b = np.log(100) + np.cumsum(rng.normal(0, 0.01, n))
    return make_series(np.exp(log_a), symbol="A"), make_series(np.exp(log_b), symbol="B")


def test_random_walk_residual_is_not_a_valid_ou_with_residual_critical_values():
    """Con el valor crítico de una serie observada (N = 1, −2,86) este par de paseos
    independientes salía «válido» con vida media de 35,7 días y sin ningún aviso. El spread es
    el residuo de una MCO estimada: se exige el valor crítico de Engle–Granger (N = 2)."""
    a, b = _random_walk_pair(1007)
    an = analyze_pair(a, b)
    ou = an.ou
    assert mackinnoncrit(N=2, regression="c", nobs=ou.nobs)[1] == ou.df_crit
    assert ou.df_crit < ou.df_stat < mackinnoncrit(N=1, regression="c", nobs=ou.nobs)[1]
    assert 0 < ou.b < 1 and ou.df_estimated_residual and ou.df_alpha == an.config.alpha_level

    assert ou.status == OUStatus.NON_STATIONARY and ou.near_unit_root and not ou.valid
    assert ou.half_life is None and ou.lam is None and ou.sigma_eq is None and ou.mu is None
    assert an.zscore is None and an.forecasts == [] and an.ou_stability is None
    assert not an.strategy_allowed
    reason = next(c for c in an.discard_reasons if c.code == "OU_NO_ESTACIONARIO")
    assert "no se puede distinguir el spread de un paseo aleatorio" in reason.message
    assert f"{ou.df_stat:.2f}".replace(".", ",") in reason.message
    assert f"{ou.df_crit:.2f}".replace(".", ",") in reason.message
    assert "RAIZ_UNITARIA_CERCANA" not in codes(an.discard_reasons)


def test_ou_stability_windows_use_the_same_unit_root_rule():
    a, b = synthetic_preset("SINT-COI")
    an = analyze_pair(a, b)
    train = an.spread.loc[: str(an.split.train_end)]
    for w in an.ou_stability.windows:
        window = train.loc[str(w.start) : str(w.end)]
        ref = fit_ou(window, estimated_residual=True, unit_root_alpha=0.05)
        assert w.valid == ref.valid and w.half_life == ref.half_life


# --------------------------------------------------------------------------- colineales


def test_collinear_pair_does_not_fit_ou_and_blocks_strategy():
    log_a, log_b = cointegrated_log_prices(n=1200, seed=1)
    a, b = series_pair(np.log(2.0) + log_b, log_b)  # A = 2·B: solo queda ruido de redondeo
    an = analyze_pair(a, b)

    assert "EG_NO_FIABLE" in codes(an.causes) and an.status != PairStatus.EVIDENCE
    assert an.ou.status == OUStatus.INSUFFICIENT and not an.ou.valid
    assert an.ou.half_life is None and an.zscore is None and an.forecasts == []
    reason = next(c for c in an.discard_reasons if c.code == "SERIES_COLINEALES")
    assert reason.status is None and "mismo activo" in reason.message
    assert "SERIES_COLINEALES" in STRATEGY_BLOCKING_CODES
    assert not an.strategy_allowed


def test_same_asset_twice_is_blocked_with_clear_cause():
    log_a, _ = cointegrated_log_prices(n=1200, seed=1)
    ko = make_series(np.exp(log_a), symbol="KO")
    an = analyze_pair(ko, make_series(np.exp(log_a), symbol=" ko "))
    assert an.status == PairStatus.INSUFFICIENT_DATA
    assert codes(an.causes) == ["MISMO_ACTIVO"]
    assert "mismo activo" in an.causes[0].message and "«KO»" in an.causes[0].message
    assert not an.strategy_allowed and an.ou is None and an.eg == {}

    # Mismo símbolo en otro proveedor: no es el mismo activo para el laboratorio, pero las
    # series idénticas se detectan como colineales y la estrategia queda bloqueada.
    other = make_series(np.exp(log_a), symbol="KO", provider="otro")
    an2 = analyze_pair(ko, other)
    assert "MISMO_ACTIVO" not in codes(an2.causes)
    assert "SERIES_COLINEALES" in codes(an2.discard_reasons) and not an2.strategy_allowed


# --------------------------------------------------------------------------- solo ilustrativo


def test_evidence_with_short_half_life_is_only_illustrative_and_banner_cites_reason():
    a, b = series_pair(*cointegrated_log_prices(n=1500, seed=4, lam=1.2, sigma=0.03))
    an = analyze_pair(a, b)
    assert an.status == PairStatus.EVIDENCE and an.ou.valid and an.ou.half_life < 1
    reason = next(c for c in an.discard_reasons if c.code == "VIDA_MEDIA_CORTA")
    assert an.strategy_allowed  # O5: se permite, pero solo como ilustración
    assert an.illustrative_only
    assert an.illustrative_reasons == [reason.message]
    banner = an.illustrative_banner
    assert banner.startswith(ILLUSTRATIVE_BANNER) and reason.message in banner
    assert banner in an.warnings


def test_evidence_with_long_half_life_is_only_illustrative():
    a, b = synthetic_preset("SINT-COI")
    an = analyze_pair(a, b, AnalysisConfig(half_life_max=3.0))
    assert an.status == PairStatus.EVIDENCE and an.ou.half_life > 3.0
    assert "VIDA_MEDIA_LARGA" in codes(an.discard_reasons)
    assert an.illustrative_only and "Vida media" in an.illustrative_banner


def test_non_evidence_banner_cites_the_status():
    an = analyze_pair(*series_pair(*random_walk_and_stationary(seed=0)))
    assert an.status == PairStatus.DOUBTFUL_ASSUMPTIONS and an.illustrative_only
    assert STATUS_LABELS[an.status] in an.illustrative_banner


# --------------------------------------------------------------------------- estabilidad


def test_flat_window_in_training_is_not_insufficient_data():
    """Un tramo plano de 300 sesiones dentro de formación hace no evaluable alguna ventana de
    estabilidad; antes eso convertía todo el par en ⛔ CALCULO_NO_POSIBLE."""
    log_a, log_b = cointegrated_log_prices(n=1500, seed=11, lam=0.1)
    log_a[300:600], log_b[300:600] = log_a[299], log_b[299]
    an = analyze_pair(*series_pair(log_a, log_b))

    assert an.status != PairStatus.INSUFFICIENT_DATA
    assert "CALCULO_NO_POSIBLE" not in codes(an.causes)
    flat = [w for w in an.stability.windows if not w.evaluable]
    assert flat and all(not w.passed and "constante" in w.reason for w in flat)
    assert flat[0].start.isoformat() in an.stability.reasons[0]
    assert an.ou is not None and an.ou.valid


@pytest.mark.parametrize(
    ("updates", "fragment"),
    [
        ({"stability_window": 20, "stability_step": 20}, "al menos 56 observaciones"),
        ({"maxlag": 0, "stability_window": 31}, "al menos 32 observaciones"),
        ({"stability_step": 0}, "paso entre ventanas"),
        ({"maxlag": -1}, "rezagos máximos"),
        ({"min_train_obs": 10}, "mínimo de observaciones en formación"),
        ({"max_fail_frac": 1.5}, "fracción máxima"),
        ({"half_life_min": 10.0, "half_life_max": 5.0}, "vida media mínima"),
        ({"correlation_min": 2.0}, "correlación mínima"),
        ({"forecast_horizons": (0, 5)}, "horizontes de pronóstico"),
        ({"forecast_horizons": ()}, "horizontes de pronóstico"),
        ({"forecast_level": 1.0}, "nivel de los intervalos"),
        ({"fdr_method": "xx"}, "método de corrección"),
    ],
)
def test_analysis_config_rejects_incoherent_parameters(updates, fragment):
    with pytest.raises(ConfigError) as exc:
        AnalysisConfig(**updates)
    assert exc.value.code == "CONFIGURACION_ANALISIS_INVALIDA"
    assert fragment in exc.value.message


def test_analysis_config_accepts_the_minimum_window():
    assert AnalysisConfig(stability_window=56).stability_window == 56
    assert AnalysisConfig(maxlag=0, stability_window=32, min_train_obs=30).maxlag == 0


# --------------------------------------------------------------------------- precios ejecutables


def _price_spread_pair_with_negative_price(index: int = 1100):
    rng = np.random.default_rng(0)
    n = 1500
    b = 20 + np.cumsum(rng.normal(0, 0.3, n))
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.9 * x[t - 1] + rng.normal(0, 0.5)
    a = 5 + b + x
    b[index] = -3.0  # como el WTI de abril de 2020
    return make_series(a, symbol="AAA"), make_series(b, symbol="BBB")


def test_non_positive_executable_price_blocks_strategy_without_raising():
    a, b = _price_spread_pair_with_negative_price()
    cfg = AnalysisConfig(spread_type=SpreadType.PRICE)
    an = analyze_pair(a, b, cfg)
    assert an.status != PairStatus.INSUFFICIENT_DATA and an.hedge is not None
    reason = next(c for c in an.discard_reasons if c.code == "PRECIO_EJECUTABLE_NO_POSITIVO")
    assert "«BBB»" in reason.message and "tramo de validación" in reason.message
    assert "precios positivos" in reason.message
    assert not an.strategy_allowed
    # run_pair no lanza: simplemente no simula.
    result = run_pair(stub_config(analysis=cfg), StubData([a, b]))
    assert result.backtests == {}


def test_index_without_proxy_requires_one_and_analysis_is_still_shown():
    a, b = synthetic_preset("SINT-COI")
    index = as_non_executable(a, suggested_proxy="SPY")
    an = analyze_pair(index, b)
    reason = next(c for c in an.discard_reasons if c.code == "PROXY_REQUERIDO")
    assert reason.status is None and "«SPY»" in reason.message
    assert "nunca se sustituye en silencio" in reason.message
    assert not an.strategy_allowed and an.exec_symbol_a is None
    assert an.status == PairStatus.EVIDENCE and an.ou.valid and an.zscore is not None
    assert an.illustrative_only

    proxy = with_symbol(a, "SPY")
    with_proxy = analyze_pair(index, b, exec_a=proxy)
    assert "PROXY_REQUERIDO" not in codes(with_proxy.discard_reasons)
    assert with_proxy.strategy_allowed and with_proxy.exec_symbol_a == "SPY"


def test_proxy_issues_are_reported_and_non_positive_proxy_price_blocks():
    a, b = synthetic_preset("SINT-COI")
    index = as_non_executable(a)
    data = a.data.copy()
    data.iloc[1000, 0] = data.iloc[999, 0] * 1.8  # salto sospechoso
    data.iloc[700, 0] = np.nan  # faltante
    data.iloc[1200, 0] = 0.0  # precio no positivo en prueba
    duplicate = data.iloc[[800]] * 3  # fecha repetida (se conserva la última)
    data = pd.concat([data.iloc[:800], duplicate, data.iloc[800:]])
    proxy = with_symbol(a.with_data(data), "SPY")
    an = analyze_pair(index, b, exec_a=proxy)

    exec_issues = [i for i in an.validation.issues if i.symbol == "SPY"]
    assert {i.code for i in exec_issues} == {
        "SALTO_SOSPECHOSO",
        "FALTANTES",
        "DUPLICADOS",
        "PRECIO_NO_POSITIVO",
    }
    assert all(i.message.startswith("Instrumento ejecutable «SPY»") for i in exec_issues)
    assert "incidencia" in an.validation.status_message
    zero_day = a.data.index[1200]
    assert zero_day in an.exec_prices.index and an.exec_prices.loc[zero_day, "a"] == 0.0
    reason = next(c for c in an.discard_reasons if c.code == "PRECIO_EJECUTABLE_NO_POSITIVO")
    assert "«SPY»" in reason.message and "tramo de prueba" in reason.message
    assert not an.strategy_allowed


def test_unknown_proxy_currency_is_unknown_currency():
    a, b = synthetic_preset("SINT-COI")
    proxy = with_symbol(a, "SPY", currency=None)
    an = analyze_pair(as_non_executable(a), b, exec_a=proxy)
    assert an.status == PairStatus.INSUFFICIENT_DATA
    cause = next(c for c in an.causes if c.code == "MONEDA_DESCONOCIDA")
    assert "desconocida" in cause.message and "None" not in cause.message


# --------------------------------------------------------------------------- calendario


def test_crypto_pair_with_exchange_proxy_uses_mixed_calendar():
    n = 2100  # días naturales
    log_a, log_b = cointegrated_log_prices(n=n, seed=11, lam=0.12, sigma=0.012)
    eth = make_series(np.exp(log_a), symbol="ETH-USD", asset_class=AssetClass.CRYPTO, freq="D")
    btc = make_series(np.exp(log_b), symbol="BTC-USD", asset_class=AssetClass.CRYPTO, freq="D")
    weekdays = btc.data["close"][btc.data.index.dayofweek < 5] * 0.001
    etf = make_series(
        weekdays.to_numpy(), symbol="BTCETF", asset_class=AssetClass.ETF, dates=weekdays.index
    )

    alone = analyze_pair(eth, btc)
    assert alone.calendar == CalendarType.CRYPTO_7D

    an = analyze_pair(eth, btc, exec_b=etf)
    assert an.calendar == CalendarType.MIXED == an.validation.calendar
    assert an.aligned.calendar == CalendarType.MIXED
    assert an.ou.time_unit == "día de negociación"
    assert "DESFASE_CIERRE_CRIPTO" in [i.code for i in an.validation.issues]
    assert any("Calendario efectivo mixto" in w for w in an.warnings)
    assert (an.exec_prices.index.dayofweek < 5).all()

    result = run_pair(
        stub_config(("ETH-USD", "BTC-USD"), exec_b="BTCETF"), StubData([eth, btc, etf])
    )
    assert result.backtests[Segment.VALIDATION].metrics.periods_per_year == 252
