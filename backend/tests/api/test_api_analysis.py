"""API de análisis: /pairs/analyze y /correlations (proveedor falso, sin red)."""

from __future__ import annotations

import json
import threading
from typing import Any

from api_support import pair_config

from mrlab import DISCLAIMER
from mrlab.analysis.pair_pipeline import ILLUSTRATIVE_BANNER

FORBIDDEN_OU_KEYS = {"half_life", "lam", "sigma_eq", "mu_se", "upper1", "lower1", "upper2"}


def strict_json(response) -> Any:
    """JSON de la respuesta; falla si contiene NaN o infinitos (JSON no estándar)."""

    def reject(token: str) -> Any:
        raise AssertionError(f"La respuesta contiene {token}, que no es JSON válido")

    return json.loads(response.text, parse_constant=reject)


def keys_in(obj: Any) -> set[str]:
    """Todas las claves de diccionarios anidados (búsqueda recursiva)."""
    found: set[str] = set()
    if isinstance(obj, dict):
        for key, value in obj.items():
            found.add(key)
            found |= keys_in(value)
    elif isinstance(obj, list):
        for item in obj:
            found |= keys_in(item)
    return found


def analyze(client, **kwargs) -> Any:
    r = client.post("/api/pairs/analyze", json=pair_config(**kwargs))
    assert r.status_code == 200, r.text
    return strict_json(r)


def codes(items: list[dict]) -> list[str]:
    return [c["code"] for c in items]


# --------------------------------------------------------------------------- /pairs/analyze


def test_cointegrated_pair_has_valid_ou_bands_and_units(client) -> None:
    body = analyze(client)
    assert body["disclaimer"] == DISCLAIMER
    assert body["synthetic"] is False and body["data_origin"] == "HISTORICO"
    an = body["analysis"]
    assert an["status"] == "evidencia"
    assert an["status_label"].startswith("✅")
    assert an["strategy_allowed"] is True and an["illustrative_only"] is False
    assert an["illustrative_banner"] is None
    ou = an["ou"]
    assert ou["status"] == "valido" and ou["valid"] is True
    assert ou["lam"] > 0 and 1 <= ou["half_life"] <= 126
    assert ou["units"]["time_unit"] == "día de negociación"
    assert ou["units"]["lam"] == "1/día de negociación"
    assert ou["units"]["half_life"] == "días de negociación"
    assert an["time_unit"] == "día de negociación" and an["periods_per_year"] == 252
    assert an["calendar"] == "bolsa"
    bands = an["bands"]
    assert bands["upper2"] > bands["upper1"] > bands["mu"] > bands["lower1"] > bands["lower2"]
    assert an["zscore"]["dates"] == an["spread"]["dates"]
    assert len(an["forecasts"]) == 3 and an["residuals"] is not None
    assert an["hedge"]["units"] and an["spread_units"] == an["hedge"]["units"]
    assert [d["direction"] for d in an["eg"]] == ["a_sobre_b", "b_sobre_a"]
    assert set(an["eg_p_adjusted"]) == {"a_sobre_b", "b_sobre_a"}
    # Configuración fijada: instantáneas y fechas de corte.
    cfg = body["config"]
    assert cfg["pair"]["a"]["snapshot_id"] and cfg["pair"]["b"]["snapshot_id"]
    assert cfg["frozen_split"]["n_test"] > 0 and cfg["include_test"] is False
    assert len(cfg["config_hash"]) == 64


def test_charts_stop_before_the_test_period(client) -> None:
    an = analyze(client)["analysis"]
    split = an["split"]
    assert an["chart_end"] == split["validation_end"]
    assert an["spread"]["dates"][-1] == split["validation_end"]
    assert an["prices"]["dates"][-1] == split["validation_end"]
    assert all(d <= split["validation_end"] for d in an["zscore"]["dates"])
    assert "prueba" in an["chart_note"] and "estudio" in an["chart_note"]


def test_invalid_ou_hides_half_life_bands_and_z(client) -> None:
    body = analyze(client, a="XOM", b="CVX")
    an = body["analysis"]
    assert an["status"] == "evidencia_insuficiente"
    assert an["ou"]["valid"] is False and an["ou"]["status"] == "no_estacionario"
    assert an["bands"] is None and an["zscore"] is None
    assert an["forecasts"] == [] and an["residuals"] is None and an["ou_stability"] is None
    assert not (keys_in(body) & FORBIDDEN_OU_KEYS), keys_in(body) & FORBIDDEN_OU_KEYS
    assert an["strategy_allowed"] is False and an["illustrative_only"] is True
    discard = {c["code"]: c["message"] for c in an["discard_reasons"]}
    assert "OU_NO_ESTACIONARIO" in discard
    assert "EVIDENCIA_INSUFICIENTE" in discard
    for message in discard.values():
        assert message and any(w in message for w in (" el ", " la ", " de ", " no "))
    text = json.dumps(body, ensure_ascii=False).lower()
    for promo in ("cointegración garantizada", "par rentable", "estrategia ganadora"):
        assert promo not in text


def test_unstable_pair_with_valid_ou_is_only_illustrative(client) -> None:
    an = analyze(client, a="BTC-USD", b="SPY")["analysis"]
    assert an["status"] == "inestable"
    assert an["ou"]["valid"] is True and an["bands"] is not None
    assert an["strategy_allowed"] is True and an["illustrative_only"] is True
    assert an["illustrative_banner"].startswith(ILLUSTRATIVE_BANNER)
    assert an["illustrative_reasons"]
    assert all(reason in an["illustrative_banner"] for reason in an["illustrative_reasons"])
    # Cripto con un ejecutable de bolsa: calendario mixto (252) y aviso del desfase de cierre.
    assert an["calendar"] == "mixto" and an["periods_per_year"] == 252
    messages = an["warnings"] + [i["message"] for i in an["validation"]["issues"]]
    assert any("medianoche" in m for m in messages)


def test_index_requires_explicit_proxy_and_is_never_substituted(client) -> None:
    body = analyze(client, a="^GSPC", b="PEP")
    an = body["analysis"]
    assert an["symbol_a"] == "^GSPC" and an["exec_symbol_a"] is None
    assert body["config"]["pair"]["exec_a"] is None
    proxy = next(c for c in an["discard_reasons"] if c["code"] == "PROXY_REQUERIDO")
    assert "SPY" in proxy["message"]
    assert an["strategy_allowed"] is False
    assert an["exec_prices"] is None


def test_index_with_chosen_proxy_trades_the_proxy(client) -> None:
    cfg = pair_config("^GSPC", "IVV")
    cfg["pair"]["exec_a"] = {**cfg["pair"]["a"], "symbol": "SPY"}
    r = client.post("/api/pairs/analyze", json=cfg)
    assert r.status_code == 200, r.text
    an = strict_json(r)["analysis"]
    assert an["exec_symbol_a"] == "SPY" and an["symbol_a"] == "^GSPC"
    assert "PROXY_REQUERIDO" not in codes(an["discard_reasons"])
    assert an["exec_prices"]["symbol_a"] == "SPY"
    r = client.post("/api/backtest", json=cfg)
    assert r.status_code == 200, r.text
    bt = strict_json(r)
    assert bt["strategy_allowed"] is True
    legs = {leg["symbol"] for leg in bt["segments"]["validacion"]["legs"].values()}
    assert legs == {"SPY", "IVV"}  # opera SPY, nunca el índice


def test_different_currencies_block_with_spanish_reason(client) -> None:
    an = analyze(client, a="KO", b="SAP.DE")["analysis"]
    assert an["status"] == "datos_insuficientes"
    assert an["status_label"].startswith("⛔")
    cause = next(c for c in an["causes"] if c["code"] == "MONEDA_DISTINTA")
    assert "moneda" in cause["message"].lower()
    assert an["strategy_allowed"] is False and an["ou"] is None


def test_include_test_is_rejected_outside_studies(client) -> None:
    r = client.post("/api/pairs/analyze", json=pair_config(include_test=True))
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["code"] == "PRUEBA_SOLO_EN_ESTUDIO"
    assert "estudio" in err["message"] and "/api/studies" in err["message"]


def test_inconsistent_thresholds_are_reported_in_spanish(client) -> None:
    r = client.post(
        "/api/pairs/analyze",
        json=pair_config(strategy={"entry_z": 1.0, "exit_z": 1.5, "stop_z": 3.0}),
    )
    assert r.status_code in (400, 422)
    err = r.json()["error"]
    assert "UMBRALES_INCONSISTENTES" in (err["code"], json.dumps(err))
    assert "salida < entrada < stop" in err["message"]


def test_malformed_body_is_spanish_422(client) -> None:
    r = client.post("/api/pairs/analyze", json={"pair": {"a": {"symbol": "KO"}}})
    assert r.status_code == 422
    err = r.json()["error"]
    assert err["code"] == "PETICION_INVALIDA"
    assert "falta este campo obligatorio" in err["message"]


def test_provider_failure_is_spanish_503(client) -> None:
    r = client.post("/api/pairs/analyze", json=pair_config(a="FALLA_RED"))
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "ERROR_RED"


def test_concurrent_analyses_share_the_cache_safely(client, fake_yahoo, app) -> None:
    """Dos análisis simultáneos del mismo par (caché vacía) no corrompen la caché."""
    results: list[Any] = [None, None]

    def worker(i: int) -> None:
        results[i] = client.post("/api/pairs/analyze", json=pair_config())

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(r.status_code == 200 for r in results), [r.text for r in results]
    first, second = (strict_json(r) for r in results)
    for key in ("pair", "frozen_split", "config_hash"):
        assert first["config"][key] == second["config"][key]
    assert first["analysis"]["ou"] == second["analysis"]["ou"]
    cache = app.state.lab.cache
    for symbol in ("KO", "PEP"):
        assert len(cache.list("yahoo", symbol)) == 1
    calls = len(fake_yahoo.calls)
    again = client.post("/api/pairs/analyze", json=pair_config())
    assert again.status_code == 200
    assert len(fake_yahoo.calls) == calls  # la tercera usa la caché, sin descargar
    assert strict_json(again)["config"]["pair"] == first["config"]["pair"]


# --------------------------------------------------------------------------- /correlations


def _ref(symbol: str) -> dict:
    return {"provider": "yahoo", "symbol": symbol, "start": "2016-01-04", "end": "2021-12-31"}


def test_correlations_matrix_and_series(client) -> None:
    r = client.post(
        "/api/correlations",
        json={"series": [_ref("KO"), _ref("PEP"), _ref("XOM")], "window": 63, "pair": [0, 2]},
    )
    assert r.status_code == 200, r.text
    body = strict_json(r)
    assert body["disclaimer"] == DISCLAIMER and body["synthetic"] is False
    assert body["symbols"] == ["KO", "PEP", "XOM"]
    for name in ("pearson", "spearman"):
        m = body[name]
        assert [m[i][i] for i in range(3)] == [1.0, 1.0, 1.0]
        assert m[0][1] == m[1][0]
    assert body["nobs"][0][1] == 1499 and isinstance(body["nobs"][0][1], int)
    assert body["pair_symbols"] == ["KO", "XOM"]
    norm = body["normalized"]
    assert all(norm["series"][s][0] == 100.0 for s in body["symbols"])
    rolling = body["rolling"]["values"]
    assert rolling[:62] == [None] * 62 and rolling[62] is not None
    assert len(body["returns"]["a"]) == len(body["returns"]["dates"]) == 1499
    assert "cointegración" in body["note"] and set(body["concepts"]) == {
        "correlacion",
        "cointegracion",
        "aptitud_ou",
    }


def test_correlations_mixed_calendar_warns(client) -> None:
    r = client.post("/api/correlations", json={"series": [_ref("BTC-USD"), _ref("SPY")]})
    assert r.status_code == 200, r.text
    body = strict_json(r)
    assert any("medianoche" in w for w in body["warnings"])
    # Intersección: solo días hábiles (nunca se rellenan fines de semana).
    assert body["nobs"][0][1] == 1499


def test_correlations_rejects_bad_pair_index(client) -> None:
    r = client.post("/api/correlations", json={"series": [_ref("KO"), _ref("PEP")], "pair": [0, 5]})
    assert r.status_code == 422
    assert "entre 0 y 1" in r.json()["error"]["message"]
    r = client.post("/api/correlations", json={"series": [_ref("KO")]})
    assert r.status_code == 422
    assert "al menos 2" in r.json()["error"]["message"]


# --------------------------------------------------------------------------- /docs


def test_openapi_documents_analysis_in_spanish(client) -> None:
    spec = client.get("/openapi.json").json()
    op = spec["paths"]["/api/pairs/analyze"]["post"]
    assert op["summary"] == "Analizar un par"
    assert "prueba" in op["description"]
    schemas = spec["components"]["schemas"]
    run_config = schemas["RunConfigIn"]["properties"]
    assert "estudio" in run_config["include_test"]["description"]
    assert "proxy" in schemas["PairSpecIn"]["properties"]["exec_a"]["description"]
    assert schemas["PairAnalysisOut"]["properties"]["bands"]["description"]
