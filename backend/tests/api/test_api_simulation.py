"""API de simulación SINTÉTICA del OU: recorte de trayectorias, histograma de cruces y límites."""

from __future__ import annotations

import math

import pytest

from mrlab import DISCLAIMER
from mrlab.api.routers.simulation import crossing_histogram
from mrlab.model.ou import MAX_SIM_HORIZON, MAX_SIM_PATHS, simulate_paths_summary

BASE = {"lam": 0.05, "mu": 0.0, "sigma": 0.02, "x0": 0.1, "horizon": 120, "seed": 42}


def simulate(client, **params):
    return client.post("/api/ou/simulate", json={**BASE, **params})


def test_simulation_is_labelled_synthetic(client):
    r = simulate(client)
    assert r.status_code == 200
    body = r.json()
    assert body["synthetic"] is True
    assert body["label"] == "SINTÉTICO"
    assert body["disclaimer"] == DISCLAIMER
    assert any("SINTÉTICOS" in n for n in body["notes"])
    assert body["time_unit"] == "día de negociación"


def test_paths_are_trimmed_but_statistics_use_all_paths(client):
    r = simulate(client, n_paths=300, paths_returned=7)
    body = r.json()
    assert body["n_paths"] == 300
    assert body["n_paths_returned"] == 7
    assert len(body["paths"]) == 7
    assert all(len(p) == BASE["horizon"] + 1 for p in body["paths"])
    assert len(body["crossing_times"]) == 300
    assert "7 de 300" in body["paths_note"]
    assert body["paths_note"] in body["notes"]
    # Las estadísticas coinciden con el núcleo calculado con las 300 trayectorias.
    core = simulate_paths_summary(
        lam=0.05, mu=0.0, sigma=0.02, x0=0.1, horizon=120, n_paths=300, seed=42
    )
    # JSON conserva los float exactos (repr), así que se comparan con igualdad.
    assert body["mean_empirical"] == core.mean_empirical
    assert body["half_life_empirical"] == core.half_life_empirical
    assert body["crossing_times"] == core.crossing_times
    assert body["n_not_crossed"] == core.n_not_crossed
    assert body["paths"] == core.paths[:7]
    # Con otro recorte, las estadísticas no cambian.
    other = simulate(client, n_paths=300, paths_returned=0).json()
    assert other["paths"] == []
    assert other["mean_empirical"] == body["mean_empirical"]
    assert other["crossing_hist"] == body["crossing_hist"]


def test_default_paths_returned_is_50(client):
    body = simulate(client, n_paths=200).json()
    assert body["n_paths_returned"] == 50
    assert len(body["paths"]) == 50


def test_paths_returned_larger_than_n_paths_is_clipped(client):
    body = simulate(client, n_paths=10, paths_returned=500).json()
    assert body["n_paths_returned"] == 10
    assert len(body["paths"]) == 10
    assert body["paths_note"] == "Se muestran las 10 trayectorias simuladas."


def test_crossing_histogram_counts_every_crossed_path(client):
    body = simulate(client, n_paths=400, horizon=60, paths_returned=0).json()
    hist = body["crossing_hist"]
    assert len(hist["bins"]) == len(hist["counts"]) + 1
    assert hist["bins"] == sorted(hist["bins"])
    crossed = [t for t in body["crossing_times"] if t is not None]
    assert sum(hist["counts"]) == len(crossed)
    assert sum(hist["counts"]) + body["n_not_crossed"] == 400
    assert hist["bins"][0] <= min(crossed) and hist["bins"][-1] >= max(crossed)
    assert len(hist["counts"]) <= 40


def test_crossing_histogram_helper():
    assert crossing_histogram([None, None]).bins == []
    h = crossing_histogram([1.0, 1.0, 2.0, None])
    assert h.bins == [0.5, 1.5, 2.5]
    assert h.counts == [2, 1]
    wide = crossing_histogram([float(t) for t in range(1, 1001)])
    assert len(wide.counts) <= 40
    assert sum(wide.counts) == 1000


def test_theoretical_quantities(client):
    body = simulate(client).json()
    assert body["sigma_eq"] == pytest.approx(0.02 / math.sqrt(0.1))
    assert body["half_life_theoretical"] == pytest.approx(math.log(2) / 0.05)
    assert body["time_grid"][0] == 0 and body["time_grid"][-1] == BASE["horizon"]
    assert body["mean_theoretical"][0] == pytest.approx(0.1)
    assert body["stationary_upper"][0] == pytest.approx(2 * 0.02 / math.sqrt(0.1))


def test_same_seed_same_paths(client):
    a = simulate(client, n_paths=20).json()
    b = simulate(client, n_paths=20).json()
    c = simulate(client, n_paths=20, seed=43).json()
    assert a["paths"] == b["paths"]
    assert a["paths"] != c["paths"]


def test_simulation_never_touches_cache(client, lab_settings):
    simulate(client, n_paths=50)
    assert client.get("/api/data/cache").json() == []
    assert not lab_settings.cache_dir.exists() or not any(lab_settings.cache_dir.rglob("*.parquet"))


def test_extreme_parameters_never_return_nan(client):
    for params in (
        {"lam": 1e-300, "sigma": 1.0, "x0": 1.0, "horizon": 5, "n_paths": 3},
        {"lam": 50.0, "sigma": 1e300, "x0": 1.0, "horizon": 5, "n_paths": 3},
        {"lam": 0.001, "sigma": 0.01, "x0": 5.0, "horizon": 3, "n_paths": 2},
    ):
        r = client.post("/api/ou/simulate", json={"mu": 0.0, "seed": 1, **params})
        assert r.status_code == 200, r.text
        assert "NaN" not in r.text and "Infinity" not in r.text


def test_no_crossing_yields_null_median_and_note(client):
    body = simulate(client, lam=0.001, sigma=0.0001, x0=5.0, horizon=3, n_paths=5).json()
    assert body["n_not_crossed"] == 5
    assert body["crossing_time_median"] is None
    assert body["crossing_hist"] == {"bins": [], "counts": []}
    assert body["half_life_empirical"] is None
    assert any("no cruzaron" in n for n in body["notes"])


# --------------------------------------------------------------------------- límites


def test_too_many_paths_is_422_in_spanish(client):
    r = simulate(client, n_paths=MAX_SIM_PATHS + 1)
    assert r.status_code == 422
    err = r.json()["error"]
    assert err["code"] == "PETICION_INVALIDA"
    assert "n_paths: debe ser menor o igual que 500" in err["message"]


def test_horizon_over_maximum_is_422(client):
    r = simulate(client, horizon=MAX_SIM_HORIZON + 1)
    assert r.status_code == 422
    assert "horizon: debe ser menor o igual que 2.520" in r.json()["error"]["message"]
    assert simulate(client, horizon=0).status_code == 422


def test_x0_equal_to_mu_is_400(client):
    r = simulate(client, x0=0.0, mu=0.0)
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["code"] == "X0_IGUAL_A_MU"
    assert "X₀ debe ser distinto de μ" in err["message"]


@pytest.mark.parametrize(
    ("params", "field"),
    [
        ({"lam": 0}, "lam"),
        ({"lam": -0.1}, "lam"),
        ({"sigma": 0}, "sigma"),
        ({"n_paths": 0}, "n_paths"),
        ({"paths_returned": 501}, "paths_returned"),
        ({"paths_returned": -1}, "paths_returned"),
        ({"seed": -1}, "seed"),
        ({"x0": "uno"}, "x0"),
        ({"otro": 1}, "otro"),
    ],
)
def test_invalid_parameters_are_422(client, params, field):
    r = simulate(client, **params)
    assert r.status_code == 422
    assert field in [c["campo"] for c in r.json()["error"]["details"]["campos"]]


def test_missing_required_fields(client):
    r = client.post("/api/ou/simulate", json={"mu": 0})
    assert r.status_code == 422
    campos = {c["campo"] for c in r.json()["error"]["details"]["campos"]}
    assert campos == {"lam", "sigma", "x0"}


@pytest.mark.parametrize(
    "raw", [b'{"lam": NaN, "sigma": 1, "x0": 1}', b'{"lam": 1, "sigma": Infinity, "x0": 1}']
)
def test_non_finite_values_rejected(client, raw):
    r = client.post("/api/ou/simulate", content=raw, headers={"content-type": "application/json"})
    assert r.status_code == 422
    assert "debe ser un número finito" in r.json()["error"]["message"]
