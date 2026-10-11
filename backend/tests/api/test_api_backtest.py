"""API de backtest: /backtest y /backtest/compare (proveedor falso, sin red)."""

from __future__ import annotations

import json
import math
from typing import Any

import pytest
from api_support import pair_config

from mrlab import DISCLAIMER
from mrlab.analysis.pair_pipeline import ILLUSTRATIVE_BANNER
from mrlab.api.dto import BLOCKED_PREFIX
from mrlab.api.routers.backtest import COMPARE_NOTE, compare_job
from mrlab.backtest.engine import DAILY_COLUMNS, EXECUTION_CONVENTION
from mrlab.backtest.metrics import METRIC_HELP
from mrlab.reporting.run_config import RunConfig


def strict_json(response) -> Any:
    def reject(token: str) -> Any:
        raise AssertionError(f"La respuesta contiene {token}, que no es JSON válido")

    return json.loads(response.text, parse_constant=reject)


def keys_in(obj: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(obj, dict):
        for key, value in obj.items():
            found.add(key)
            found |= keys_in(value)
    elif isinstance(obj, list):
        for item in obj:
            found |= keys_in(item)
    return found


def backtest(client, **kwargs) -> Any:
    r = client.post("/api/backtest", json=pair_config(**kwargs))
    assert r.status_code == 200, r.text
    return strict_json(r)


# --------------------------------------------------------------------------- /backtest


def test_backtest_has_train_and_validation_never_test(client) -> None:
    body = backtest(client)
    assert body["disclaimer"] == DISCLAIMER and body["synthetic"] is False
    assert body["execution_convention"] == EXECUTION_CONVENTION
    assert body["status"] == "evidencia" and body["strategy_allowed"] is True
    assert body["blocked_reason"] is None
    assert list(body["segments"]) == ["formacion", "validacion"]
    assert body["test_included"] is False and body["test_contaminated"] is None
    train, val = body["segments"]["formacion"], body["segments"]["validacion"]
    assert train["in_sample"] is True and val["in_sample"] is False
    assert train["label"] == "formación" and val["label"] == "validación"
    assert any("DENTRO DE MUESTRA" in w for w in train["warnings"])
    split = body["config"]["frozen_split"]
    assert (train["start"], train["end"]) == (split["train_start"], split["train_end"])
    assert (val["start"], val["end"]) == (split["validation_start"], split["validation_end"])
    assert "dentro de muestra" in body["in_sample_note"].lower()


def test_backtest_segment_payload(client) -> None:
    val = backtest(client)["segments"]["validacion"]
    columns = val["daily"]["columns"]
    assert list(columns) == DAILY_COLUMNS
    lengths = {len(v) for v in columns.values()}
    assert lengths == {len(columns["date"])}
    assert val["metric_help"] == METRIC_HELP
    metrics = val["metrics"]
    assert set(METRIC_HELP) == set(metrics)  # una ayuda de una línea por métrica
    assert metrics["periods_per_year"] == val["periods_per_year"] == 252
    assert "rf = 0" in metrics["sharpe_convention"]
    assert val["legs"]["y"]["symbol"] in ("KO", "PEP") and val["legs"]["x"]["symbol"] in (
        "KO",
        "PEP",
    )
    assert val["spread_type"] == "log" and val["calendar"] == "bolsa"
    assert val["max_holding_bars"] >= 5 and isinstance(val["beta"], float)
    assert val["trades"], "con este par la validación debe tener operaciones"
    trade = val["trades"][0]
    assert {"entry_signal_date", "entry_date", "net_pnl", "total_costs"} <= set(trade)
    assert trade["entry_date"] > trade["entry_signal_date"]  # ejecución en t+1


def test_backtest_accounting_is_consistent(client) -> None:
    body = backtest(client)
    capital = body["config"]["strategy"]["initial_capital"]
    for segment in body["segments"].values():
        cols = segment["daily"]["columns"]
        m = segment["metrics"]
        pnl_net = sum(cols["pnl_net"])
        assert cols["equity"][-1] == pytest.approx(capital + pnl_net, rel=1e-9, abs=1e-6)
        assert m["final_equity"] == pytest.approx(cols["equity"][-1], rel=1e-12)
        assert m["net_pnl"] == pytest.approx(pnl_net, abs=1e-6)
        for gross, cost, net in zip(
            cols["pnl_gross"], cols["cost_total"], cols["pnl_net"], strict=True
        ):
            assert gross - cost == pytest.approx(net, abs=1e-9)
        assert m["gross_pnl"] - m["total_costs"] == pytest.approx(m["net_pnl"], abs=1e-6)
        assert m["realized_pnl"] + m["unrealized_pnl"] == pytest.approx(m["net_pnl"], abs=1e-6)
        assert sum(t["net_pnl"] for t in segment["trades"]) == pytest.approx(m["net_pnl"], abs=1e-6)


def test_blocked_strategy_returns_reason_not_error(client) -> None:
    body = backtest(client, a="XOM", b="CVX")
    assert body["strategy_allowed"] is False and body["segments"] == {}
    assert body["blocked_reason"].startswith(BLOCKED_PREFIX)
    assert "OU" in body["blocked_reason"] or "Ornstein" in body["blocked_reason"]
    assert "OU_NO_ESTACIONARIO" in [c["code"] for c in body["discard_reasons"]]
    assert not keys_in(body) & {"half_life", "lam", "sigma_eq"}


def test_illustrative_banner_heads_each_segment(client) -> None:
    body = backtest(client, a="BTC-USD", b="SPY")
    assert body["illustrative_only"] is True and body["strategy_allowed"] is True
    assert body["illustrative_banner"].startswith(ILLUSTRATIVE_BANNER)
    for segment in body["segments"].values():
        assert segment["warnings"][0] == body["illustrative_banner"]
        assert segment["calendar"] == "mixto"


def test_backtest_rejects_include_test(client) -> None:
    r = client.post("/api/backtest", json=pair_config(include_test=True))
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["code"] == "PRUEBA_SOLO_EN_ESTUDIO"
    assert "una sola vez" in err["message"]


def test_backtest_currency_mismatch_is_blocked(client) -> None:
    body = backtest(client, a="KO", b="SAP.DE")
    assert body["status"] == "datos_insuficientes" and body["segments"] == {}
    assert "moneda" in body["blocked_reason"].lower()


# --------------------------------------------------------------------------- /backtest/compare


def _compare_configs() -> list[dict]:
    configs = [
        pair_config("XOM", "CVX"),
        pair_config("KO", "PEP"),
        pair_config("KO", "SAP.DE"),
        pair_config("FALLA_RED", "PEP"),
        pair_config("BTC-USD", "SPY", strategy={"initial_capital": 5000.0}),
    ]
    return configs


def test_compare_runs_in_background_and_keeps_input_order(client, app) -> None:
    r = client.post("/api/backtest/compare", json={"configs": _compare_configs()})
    assert r.status_code == 202, r.text
    accepted = r.json()
    job_id = accepted["job_id"]
    assert accepted["status_url"] == f"/api/jobs/{job_id}"
    job = app.state.lab.jobs.wait(job_id, timeout=120)
    assert job.status == "completado", job.error
    r = client.get(f"/api/jobs/{job_id}")
    assert r.status_code == 200
    body = strict_json(r)
    assert body["progress"] == 1.0
    result = body["result"]
    assert result["disclaimer"] == DISCLAIMER and result["note"] == COMPARE_NOTE
    items = result["items"]
    assert [it["index"] for it in items] == [0, 1, 2, 3, 4]
    assert [it["symbol_a"] for it in items] == ["XOM", "KO", "KO", "FALLA_RED", "BTC-USD"]
    xom, ko, sap, net_fail, btc = items
    assert xom["strategy_allowed"] is False and xom["validation_metrics"] is None
    assert xom["blocked_reason"].startswith(BLOCKED_PREFIX) and xom["error"] is None
    assert ko["status"] == "evidencia" and ko["validation_metrics"]["n_observations"] > 0
    # Monedas distintas: bloqueado (⛔), pero no es un error ni tumba el trabajo.
    assert sap["status"] == "datos_insuficientes" and sap["error"] is None
    assert "moneda" in sap["blocked_reason"].lower()
    # Un error del proveedor queda en su fila, en español; los demás pares siguen.
    assert net_fail["error"]["code"] == "ERROR_RED" and net_fail["status"] is None
    assert net_fail["error"]["message"]
    # Cada par con su propio capital.
    assert btc["initial_capital"] == 5000.0
    assert btc["validation_metrics"]["initial_capital"] == 5000.0
    assert ko["initial_capital"] == 10000.0
    assert btc["illustrative_banner"].startswith(ILLUSTRATIVE_BANNER)
    assert ko["config"]["pair"]["a"]["snapshot_id"]


def test_compare_never_sorts_by_pnl(client, app) -> None:
    configs = [pair_config("BTC-USD", "SPY"), pair_config("KO", "PEP"), pair_config("SPY", "IVV")]
    r = client.post("/api/backtest/compare", json={"configs": configs})
    job = app.state.lab.jobs.wait(r.json()["job_id"], timeout=120)
    items = job.result["items"]
    labels = [it["label"] for it in items]
    assert labels == ["BTC-USD / SPY", "KO / PEP", "SPY / IVV"]
    pnls = [it["validation_metrics"]["net_pnl"] for it in items]
    assert all(math.isfinite(p) for p in pnls)


def test_compare_reports_progress_per_pair(app) -> None:
    data = app.state.lab.data
    configs = [RunConfig.model_validate(c) for c in _compare_configs()[:3]]
    calls: list[tuple[float, str]] = []
    result = compare_job(configs, data)(lambda p, m="": calls.append((p, m)))
    assert [round(p, 4) for p, _ in calls] == [0.0, round(1 / 3, 4), round(2 / 3, 4), 1.0]
    assert calls[0][1] == "Analizando el par 1 de 3: XOM / CVX."
    assert "terminada" in calls[-1][1]
    assert [it.index for it in result.items] == [0, 1, 2]


def test_compare_validates_before_creating_the_job(client, app) -> None:
    configs = [pair_config(), pair_config("SPY", "IVV", include_test=True)]
    r = client.post("/api/backtest/compare", json={"configs": configs})
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["code"] == "PRUEBA_SOLO_EN_ESTUDIO" and err["details"]["indice"] == 1
    assert app.state.lab.jobs.list() == []
    r = client.post("/api/backtest/compare", json={"configs": [pair_config()]})
    assert r.status_code == 422
    assert "al menos 2" in r.json()["error"]["message"]


def test_compare_label_names_the_executable_proxy(client, app) -> None:
    cfg = pair_config("^GSPC", "IVV")
    cfg["pair"]["exec_a"] = {**cfg["pair"]["a"], "symbol": "SPY"}
    r = client.post("/api/backtest/compare", json={"configs": [cfg, pair_config()]})
    job = app.state.lab.jobs.wait(r.json()["job_id"], timeout=120)
    first = job.result["items"][0]
    assert first["label"] == "^GSPC (opera SPY) / IVV"
    assert first["strategy_allowed"] is True


def test_openapi_documents_backtest(client) -> None:
    spec = client.get("/openapi.json").json()
    compare = spec["paths"]["/api/backtest/compare"]["post"]
    assert "202" in compare["responses"]
    assert "PnL" in compare["description"]
    seg = spec["components"]["schemas"]["BacktestSegmentOut"]["properties"]
    assert "formación" in seg["in_sample"]["description"].lower()


def test_config_from_a_response_can_be_sent_back(client, fake_yahoo) -> None:
    """La interfaz reenvía la configuración fijada que devuelve /pairs/analyze."""
    analyzed = strict_json(client.post("/api/pairs/analyze", json=pair_config()))
    calls = len(fake_yahoo.calls)
    r = client.post("/api/backtest", json=analyzed["config"])
    assert r.status_code == 200, r.text
    body = strict_json(r)
    assert body["config"]["config_hash"] == analyzed["config"]["config_hash"]
    assert body["config"]["frozen_split"] == analyzed["config"]["frozen_split"]
    assert len(fake_yahoo.calls) == calls  # instantáneas fijadas: sin descargas
