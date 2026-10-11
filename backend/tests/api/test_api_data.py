"""API de datos: carga con caché, CSV (límite de tamaño, marca SINTÉTICA), sintéticos, caché,
series y validación. Sin red: el proveedor «yahoo» es ``FakeYahoo``."""

from __future__ import annotations

import io
import threading

import pandas as pd
import pytest
from api_support import FakeYahoo
from fastapi.testclient import TestClient

from mrlab import DISCLAIMER
from mrlab.api.main import create_app
from mrlab.api.routers.data import MULTIPART_OVERHEAD, max_upload_bytes, read_limited
from mrlab.data.providers.csv_provider import CSVProvider
from mrlab.data.providers.synthetic import SyntheticProvider
from mrlab.data.registry import ProviderRegistry

RANGE = {"start": "2016-01-04", "end": "2020-12-31"}
CSV_FORM = {"symbol": "MIX", "currency": "usd", "asset_class": "acción", "price_type": "cierre"}


def load(client, symbols, provider=None, **extra):
    body = {"symbols": symbols, **RANGE, **extra}
    if provider:
        body["provider"] = provider
    return client.post("/api/data/load", json=body)


def csv_text(n: int = 30, extra_col: str | None = None, start: str = "2020-01-01") -> str:
    dates = pd.bdate_range(start, periods=n)
    header = "date,close" + (",tipo_datos" if extra_col else "")
    rows = [
        f"{d.date().isoformat()},{100 + i * 0.5}" + (f",{extra_col}" if extra_col else "")
        for i, d in enumerate(dates)
    ]
    return header + "\n" + "\n".join(rows) + "\n"


def upload(client, content, filename="mix.csv", **form):
    return client.post(
        "/api/data/csv",
        files={"file": (filename, content, "text/csv")},
        data={**CSV_FORM, **form},
    )


# --------------------------------------------------------------------------- /data/load


def test_load_downloads_and_returns_snapshot_ids(client, fake_yahoo: FakeYahoo):
    r = load(client, ["KO", "PEP"])
    assert r.status_code == 200
    body = r.json()
    assert body["disclaimer"] == DISCLAIMER
    assert body["synthetic"] is False
    assert body["errors"] == []
    assert body["cached"] == []
    assert [m["asset"]["symbol"] for m in body["series"]] == ["KO", "PEP"]
    for meta in body["series"]:
        assert meta["snapshot_id"].startswith(f"yahoo_{meta['asset']['symbol']}_1d_")
        assert meta["requested_start"] == RANGE["start"]
        assert meta["requested_end"] == RANGE["end"]
        assert meta["n_obs"] > 1000
    assert [c[0] for c in fake_yahoo.calls] == ["KO", "PEP"]


def test_second_load_uses_cache_without_calling_provider(client, fake_yahoo: FakeYahoo):
    first = load(client, ["KO", "PEP"]).json()
    calls = len(fake_yahoo.calls)
    second = load(client, ["KO", "PEP"]).json()
    assert len(fake_yahoo.calls) == calls  # ni una descarga más
    assert second["cached"] == ["KO", "PEP"]
    assert [m["snapshot_id"] for m in second["series"]] == [
        m["snapshot_id"] for m in first["series"]
    ]
    # Un subrango también sale de la caché (la instantánea completa lo cubre).
    sub = client.post(
        "/api/data/load", json={"symbols": ["ko"], "start": "2017-01-02", "end": "2018-12-31"}
    ).json()
    assert len(fake_yahoo.calls) == calls
    assert sub["cached"] == ["ko"]
    meta = sub["series"][0]
    assert meta["snapshot_id"] == first["series"][0]["snapshot_id"]
    assert meta["coverage_start"] >= "2017-01-02"
    assert meta["coverage_end"] <= "2018-12-31"


def test_refresh_forces_download(client, fake_yahoo: FakeYahoo):
    load(client, ["KO"])
    assert len(fake_yahoo.calls) == 1
    r = load(client, ["KO"], refresh=True)
    assert r.status_code == 200
    assert len(fake_yahoo.calls) == 2
    assert r.json()["cached"] == []


def test_wider_range_than_cache_downloads_again(client, fake_yahoo: FakeYahoo):
    client.post(
        "/api/data/load", json={"symbols": ["KO"], "start": "2017-01-02", "end": "2018-12-31"}
    )
    load(client, ["KO"])
    assert len(fake_yahoo.calls) == 2


def test_partial_failures_are_reported_per_symbol(client):
    r = load(client, ["KO", "NOPE", "FALLA_RED", "LIMITE"])
    assert r.status_code == 200
    body = r.json()
    assert [m["asset"]["symbol"] for m in body["series"]] == ["KO"]
    errors = {e["symbol"]: e for e in body["errors"]}
    assert errors["NOPE"]["status"] == 404
    assert errors["NOPE"]["error"]["code"] == "SIMBOLO_SIN_DATOS"
    assert errors["FALLA_RED"]["status"] == 503
    assert errors["FALLA_RED"]["error"]["code"] == "ERROR_RED"
    assert errors["LIMITE"]["status"] == 429
    assert errors["LIMITE"]["error"]["code"] == "LIMITE_PETICIONES"
    for e in errors.values():
        assert e["error"]["message"]  # en español, del proveedor
        assert set(e["error"]) == {"code", "message", "details"}


@pytest.mark.parametrize(
    ("symbol", "status", "code"),
    [
        ("FALLA_RED", 503, "ERROR_RED"),
        ("LIMITE", 429, "LIMITE_PETICIONES"),
        ("NOPE", 404, "SIMBOLO_SIN_DATOS"),
    ],
)
def test_all_failed_returns_first_error_with_http_status(client, symbol, status, code):
    r = load(client, [symbol])
    assert r.status_code == status
    err = r.json()["error"]
    assert err["code"] == code
    assert "Yahoo" in err["message"]
    if status == 429:
        assert r.headers["retry-after"] == "60"
    else:
        assert "retry-after" not in r.headers


def test_all_failed_uses_the_first_error(client):
    r = load(client, ["LIMITE", "FALLA_RED"])
    assert r.status_code == 429
    assert r.json()["error"]["code"] == "LIMITE_PETICIONES"


def test_load_unknown_provider_is_404(client):
    r = load(client, ["KO"], provider="tiingo")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "PROVEEDOR_DESCONOCIDO"


def test_load_index_is_never_replaced_by_proxy(client, fake_yahoo: FakeYahoo):
    r = load(client, ["^GSPC"])
    assert r.status_code == 200
    body = r.json()
    meta = body["series"][0]
    assert meta["asset"]["symbol"] == "^GSPC"
    assert meta["asset"]["executable"] is False
    assert meta["asset"]["suggested_proxy"] == "SPY"
    assert [c[0] for c in fake_yahoo.calls] == ["^GSPC"]  # no se descargó SPY
    assert any("SPY" in n and "no se aplica automáticamente" in n for n in body["notices"])


def test_load_synthetic_provider_is_flagged(client):
    r = load(client, ["SINT-COI-A", "SINT-COI-B"], provider="synthetic")
    assert r.status_code == 200
    body = r.json()
    assert body["synthetic"] is True
    assert all(m["asset"]["origin"] == "SINTETICO" for m in body["series"])
    assert any("SINTÉTICA" in n for n in body["notices"])


@pytest.mark.parametrize(
    ("body", "field"),
    [
        ({"symbols": [], **RANGE}, "symbols"),
        ({"symbols": ["KO", " ko "], **RANGE}, "symbols"),
        ({"symbols": ["KO", "  "], **RANGE}, "symbols"),
        ({"symbols": [f"S{i}" for i in range(31)], **RANGE}, "symbols"),
        (
            {"symbols": ["KO"], "start": "2021-01-01", "end": "2020-01-01"},
            "(cuerpo de la petición)",
        ),
        ({"symbols": ["KO"], "start": "2020-02-30", "end": "2020-03-01"}, "start"),
        ({"symbols": ["KO"], **RANGE, "refresh": "quizás"}, "refresh"),
        ({"symbols": ["KO"], **RANGE, "extra": 1}, "extra"),
        ({"symbols": "KO", **RANGE}, "symbols"),
    ],
)
def test_load_validation_errors_in_spanish(client, body, field):
    r = client.post("/api/data/load", json=body)
    assert r.status_code == 422
    err = r.json()["error"]
    assert err["code"] == "PETICION_INVALIDA"
    assert err["message"].startswith("La petición tiene datos no válidos.")
    assert field in [c["campo"] for c in err["details"]["campos"]]


def test_load_duplicate_symbol_message(client):
    r = client.post("/api/data/load", json={"symbols": ["KO", "ko"], **RANGE})
    assert "repetido" in r.json()["error"]["message"]


# --------------------------------------------------------------------------- /data/csv


def test_csv_upload_registers_and_caches(client):
    r = upload(client, csv_text(40))
    assert r.status_code == 200, r.text
    body = r.json()
    meta = body["series"]
    assert meta["asset"]["symbol"] == "MIX"
    assert meta["asset"]["currency"] == "USD"
    assert meta["asset"]["asset_class"] == "accion"
    assert meta["asset"]["provider"] == "csv"
    assert meta["n_obs"] == 40
    assert body["synthetic"] is False
    assert body["synthetic_forced"] is False
    assert body["notices"] == []
    assert body["disclaimer"] == DISCLAIMER
    sid = meta["snapshot_id"]
    cached = client.get("/api/data/cache", params={"provider": "csv"}).json()
    assert [m["snapshot_id"] for m in cached] == [sid]
    # La serie se puede leer de la caché y cargar por /data/load con el proveedor «csv».
    series = client.get("/api/data/series", params={"snapshot_id": sid}).json()
    assert len(series["close"]["dates"]) == 40
    # Con el mismo rango (la cobertura del archivo) es la misma instantánea; el rango pedido
    # forma parte de su identidad (decisión A9), así que otro rango da otro id.
    same = {"provider": "csv", "symbols": ["mix"], "start": "2020-01-01"}
    r = client.post("/api/data/load", json={**same, "end": meta["coverage_end"]})
    assert r.status_code == 200
    assert r.json()["series"][0]["snapshot_id"] == sid
    r = client.post("/api/data/load", json={**same, "end": "2020-12-31"})
    assert r.status_code == 200
    assert r.json()["series"][0]["n_obs"] == 40


def test_csv_reupload_replaces_series_with_new_snapshot(client):
    first = upload(client, csv_text(30)).json()["series"]["snapshot_id"]
    second_meta = upload(client, csv_text(35)).json()["series"]
    second = second_meta["snapshot_id"]
    assert first != second
    r = client.post(
        "/api/data/load",
        json={
            "provider": "csv",
            "symbols": ["MIX"],
            "start": second_meta["coverage_start"],
            "end": second_meta["coverage_end"],
        },
    )
    assert r.json()["series"][0]["snapshot_id"] == second
    assert r.json()["series"][0]["n_obs"] == 35  # el archivo nuevo, no el anterior
    # La instantánea anterior sigue disponible para reproducir análisis antiguos.
    assert client.get("/api/data/series", params={"snapshot_id": first}).status_code == 200


def test_csv_with_synthetic_column_is_forced_synthetic(client):
    r = upload(client, csv_text(30, extra_col="SINTETICO"), origin="HISTORICO")
    assert r.status_code == 200, r.text
    body = r.json()
    meta = body["series"]
    assert body["synthetic"] is True
    assert body["synthetic_forced"] is True
    assert meta["asset"]["origin"] == "SINTETICO"
    assert meta["price_type"] == "sintetico"
    assert meta["asset"]["name"].startswith("SINTÉTICO")
    assert any("formulario declaraba datos históricos" in n for n in body["notices"])
    assert any("tipo_datos" in n for n in body["notices"])
    # La marca viaja con la serie en la caché.
    series = client.get("/api/data/series", params={"snapshot_id": meta["snapshot_id"]}).json()
    assert series["synthetic"] is True
    assert series["meta"]["asset"]["origin"] == "SINTETICO"


def test_csv_with_synthetic_filename_is_forced_synthetic(client):
    r = upload(client, csv_text(30), filename="SINTETICO_mix.csv")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["synthetic"] is True
    assert body["synthetic_forced"] is True
    assert any("SINTETICO_mix.csv" in n for n in body["notices"])


def test_csv_declared_synthetic_is_not_reported_as_forced(client):
    body = upload(client, csv_text(30), origin="SINTETICO").json()
    assert body["synthetic"] is True
    assert body["synthetic_forced"] is False
    assert body["notices"] == [
        "Datos SINTÉTICOS generados por el laboratorio: no son precios históricos."
    ]
    seeded = upload(client, csv_text(30), symbol="SEM", seed="11").json()
    assert seeded["synthetic"] is True
    assert seeded["series"]["seed"] == 11


def test_csv_historical_column_never_downgrades_declared_synthetic(client):
    body = upload(client, csv_text(30, extra_col="HISTORICO"), origin="SINTETICO").json()
    assert body["synthetic"] is True
    assert body["series"]["asset"]["origin"] == "SINTETICO"


@pytest.mark.parametrize(
    ("form", "code"),
    [
        ({"currency": "dolares"}, "CSV_METADATOS_INVALIDOS"),
        ({"asset_class": "bono"}, "CSV_METADATOS_INVALIDOS"),
        ({"price_type": "medio"}, "CSV_METADATOS_INVALIDOS"),
        ({"seed": "abc"}, "CSV_METADATOS_INVALIDOS"),
    ],
)
def test_csv_invalid_metadata_is_422_in_spanish(client, form, code):
    r = upload(client, csv_text(10), **form)
    assert r.status_code == 422
    err = r.json()["error"]
    assert err["code"] == code
    assert "no son válidos" in err["message"]


@pytest.mark.parametrize(
    ("content", "code"),
    [
        ("date,close\n", "CSV_VACIO"),
        ("fecha;volumen\n2020-01-01;3\n", "CSV_FALTA_COLUMNA"),
        ("date,close\n2020-31-01,1\n", "CSV_FECHA_INVALIDA"),
        ("date,close\n2020-01-01,uno\n", "CSV_VALOR_NO_NUMERICO"),
        ("date,close,tipo_datos\n2020-01-01,1,REAL\n", "CSV_TIPO_DATOS_DESCONOCIDO"),
    ],
)
def test_csv_content_errors_are_422(client, content, code):
    r = upload(client, content)
    assert r.status_code == 422
    assert r.json()["error"]["code"] == code
    assert client.get("/api/data/cache", params={"provider": "csv"}).json() == []


def test_csv_missing_fields_are_listed(client):
    r = client.post("/api/data/csv", data={"symbol": "X"})
    assert r.status_code == 422
    campos = {c["campo"] for c in r.json()["error"]["details"]["campos"]}
    assert {"file", "currency", "asset_class", "price_type"} <= campos


def _too_large_body(r):
    assert r.status_code == 413
    err = r.json()["error"]
    assert err["code"] == "ARCHIVO_DEMASIADO_GRANDE"
    assert "1 MB" in err["message"]
    assert err["details"]["max_bytes"] == 1024 * 1024


def test_csv_over_limit_rejected_by_content_length(client, lab_settings):
    big = b"date,close\n" + b"2020-01-01,1\n" * 200_000  # ≈ 2,6 MB > 1 MB
    _too_large_body(upload(client, big))
    assert client.get("/api/data/cache").json() == []
    assert not list(lab_settings.uploads_dir.glob("*.csv"))  # no se registró nada


def test_csv_just_over_limit_rejected_when_reading(client, lab_settings):
    limit = max_upload_bytes(lab_settings)
    header = b"date,close\n"
    content = header + b"2020-01-01,1\n" * ((limit - len(header)) // 13 + 1)
    assert limit < len(content) < limit + MULTIPART_OVERHEAD  # pasa el filtro de la cabecera
    _too_large_body(upload(client, content))


def test_csv_at_limit_is_accepted(client, lab_settings):
    limit = max_upload_bytes(lab_settings)
    dates = pd.bdate_range("1990-01-01", periods=60_000)
    lines = [f"{d.date().isoformat()},{100 + (i % 7)}" for i, d in enumerate(dates)]
    content = ("date,close\n" + "\n".join(lines) + "\n").encode()
    content = content[: content.rfind(b"\n", 0, limit) + 1]
    assert len(content) <= limit
    r = upload(client, content)
    assert r.status_code == 200, r.text


def test_csv_chunked_body_over_limit_is_cut(client):
    """Sin Content-Length (transferencia por bloques): se corta al superar el límite."""

    def body():
        yield (
            b'--frontera\r\nContent-Disposition: form-data; name="file"; filename="a.csv"\r\n'
            b"Content-Type: text/csv\r\n\r\ndate,close\r\n"
        )
        for _ in range(400):
            yield b"2020-01-01,1\r\n" * 400

    r = client.post(
        "/api/data/csv",
        content=body(),
        headers={"content-type": "multipart/form-data; boundary=frontera"},
    )
    assert "content-length" not in {k.lower() for k in r.request.headers}
    _too_large_body(r)


def test_read_limited_reads_in_chunks_and_stops(lab_settings):
    class Counting(io.BytesIO):
        reads = 0

        def read(self, size=-1):
            Counting.reads += 1
            assert size > 0  # nunca «leer todo»
            return super().read(size)

    data = b"x" * (3 * 1024 * 1024)
    stream = Counting(data)
    with pytest.raises(Exception) as exc:
        read_limited(stream, 1024 * 1024, lab_settings)
    assert getattr(exc.value, "status_code", None) == 413
    assert stream.tell() <= 1024 * 1024 + 64 * 1024  # dejó de leer al pasar el límite
    assert read_limited(io.BytesIO(b"abc"), 10, lab_settings) == b"abc"


def test_csv_disabled_provider(lab_settings, fake_yahoo):
    registry = ProviderRegistry({"yahoo": fake_yahoo}, default_name="yahoo")
    with TestClient(create_app(lab_settings, registry), raise_server_exceptions=False) as c:
        r = upload(c, csv_text(10))
    assert r.status_code == 404
    err = r.json()["error"]
    assert err["code"] == "PROVEEDOR_DESCONOCIDO"
    assert "CSV" in err["message"]


# --------------------------------------------------------------------------- /data/synthetic


def test_synthetic_pair_is_cached_and_labelled(client):
    r = client.post("/api/data/synthetic", json={"scenario": "cointegrado", "seed": 3})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["synthetic"] is True
    assert body["label"] == "SINTÉTICO"
    assert body["disclaimer"] == DISCLAIMER
    assert body["scenario"] == "cointegrado"
    assert body["params"]["seed"] == 3
    assert "semilla 3" in body["note"]
    for leg in ("a", "b"):
        meta = body[leg]
        assert meta["asset"]["origin"] == "SINTETICO"
        assert meta["asset"]["name"].startswith("SINTÉTICO")
        assert meta["price_type"] == "sintetico"
        assert meta["seed"] == 3
        assert meta["retrieved_at"] is None
        assert meta["n_obs"] == 1500
        series = client.get("/api/data/series", params={"snapshot_id": meta["snapshot_id"]})
        assert series.status_code == 200
        assert series.json()["synthetic"] is True
    cached = {
        m["snapshot_id"]
        for m in client.get("/api/data/cache", params={"provider": "synthetic"}).json()
    }
    assert {body["a"]["snapshot_id"], body["b"]["snapshot_id"]} <= cached


def test_synthetic_is_reproducible(client):
    req = {"scenario": "correlado_no_cointegrado", "seed": 5, "n_obs": 600, "rho": 0.8}
    first = client.post("/api/data/synthetic", json=req).json()
    second = client.post("/api/data/synthetic", json=req).json()
    for leg in ("a", "b"):
        assert first[leg]["snapshot_id"] == second[leg]["snapshot_id"]
        assert first[leg]["asset"]["symbol"] == second[leg]["asset"]["symbol"]
    other = client.post("/api/data/synthetic", json={**req, "seed": 6}).json()
    assert other["a"]["snapshot_id"] != first["a"]["snapshot_id"]


def test_synthetic_scenario_defaults_apply_to_unset_fields(client):
    body = client.post(
        "/api/data/synthetic", json={"scenario": "cointegrado_baja_correlacion", "n_obs": 300}
    ).json()
    assert body["params"]["ou_lambda"] == 0.5
    assert body["params"]["ou_sigma"] == 0.03
    explicit = client.post(
        "/api/data/synthetic",
        json={"scenario": "cointegrado_baja_correlacion", "n_obs": 300, "ou_lambda": 0.2},
    ).json()
    assert explicit["params"]["ou_lambda"] == 0.2
    assert explicit["params"]["ou_sigma"] == 0.03


def test_synthetic_survives_restart_through_cache(lab_settings, fake_yahoo, client):
    body = client.post("/api/data/synthetic", json={"seed": 9, "n_obs": 200}).json()
    sid = body["a"]["snapshot_id"]
    # Proceso nuevo: el registro en memoria del generador se pierde, la caché no.
    registry = ProviderRegistry.from_toml(
        lab_settings.providers_file,
        overrides={
            "yahoo": fake_yahoo,
            "csv": CSVProvider(lab_settings.uploads_dir),
            "synthetic": SyntheticProvider(),
        },
    )
    with TestClient(create_app(lab_settings, registry)) as c2:
        r = c2.get("/api/data/series", params={"snapshot_id": sid})
    assert r.status_code == 200
    assert r.json()["synthetic"] is True


@pytest.mark.parametrize(
    ("body", "field"),
    [
        ({"scenario": "otro"}, "scenario"),
        ({"n_obs": 1}, "n_obs"),
        ({"n_obs": 20_001}, "n_obs"),
        ({"rho": 1.5}, "rho"),
        ({"ou_lambda": 0}, "ou_lambda"),
        ({"seed": -1}, "seed"),
        ({"b_start_price": -5}, "b_start_price"),
        ({"desconocido": 1}, "desconocido"),
    ],
)
def test_synthetic_invalid_params(client, body, field):
    r = client.post("/api/data/synthetic", json=body)
    assert r.status_code == 422
    assert field in [c["campo"] for c in r.json()["error"]["details"]["campos"]]


def test_synthetic_non_finite_params_rejected(client):
    r = client.post(
        "/api/data/synthetic",
        content=b'{"alpha": NaN, "beta": Infinity}',
        headers={"content-type": "application/json"},
    )
    assert r.status_code == 422
    campos = {c["campo"]: c["motivo"] for c in r.json()["error"]["details"]["campos"]}
    assert campos == {"alpha": "debe ser un número finito", "beta": "debe ser un número finito"}


# --------------------------------------------------------------------------- caché y series


def test_cache_listing_filters_and_never_downloads(client, fake_yahoo: FakeYahoo):
    assert client.get("/api/data/cache").json() == []
    load(client, ["KO", "PEP"])
    calls = len(fake_yahoo.calls)
    everything = client.get("/api/data/cache").json()
    assert {m["asset"]["symbol"] for m in everything} == {"KO", "PEP"}
    only_ko = client.get("/api/data/cache", params={"provider": "yahoo", "symbol": " ko "}).json()
    assert [m["asset"]["symbol"] for m in only_ko] == ["KO"]
    assert client.get("/api/data/cache", params={"provider": "csv"}).json() == []
    assert client.get("/api/data/cache", params={"symbol": "XOM"}).json() == []
    assert len(fake_yahoo.calls) == calls


def test_series_returns_cached_prices_without_filling_dates(client, fake_yahoo: FakeYahoo):
    meta = load(client, ["BTC-USD", "KO"]).json()["series"]
    calls = len(fake_yahoo.calls)
    ko = client.get("/api/data/series", params={"snapshot_id": meta[1]["snapshot_id"]}).json()
    dates = pd.DatetimeIndex(ko["close"]["dates"])
    assert (dates.dayofweek < 5).all()  # bolsa: nunca se añaden fines de semana
    assert len(dates) == meta[1]["n_obs"]
    assert ko["meta"]["asset"]["symbol"] == "KO"
    assert ko["synthetic"] is False
    assert ko["disclaimer"] == DISCLAIMER
    btc = client.get("/api/data/series", params={"snapshot_id": meta[0]["snapshot_id"]}).json()
    assert (pd.DatetimeIndex(btc["close"]["dates"]).dayofweek >= 5).any()  # cripto: 7 días
    trimmed = client.get(
        "/api/data/series",
        params={"snapshot_id": meta[1]["snapshot_id"], "start": "2018-01-01", "end": "2018-01-31"},
    ).json()
    assert trimmed["close"]["dates"][0] >= "2018-01-01"
    assert trimmed["close"]["dates"][-1] <= "2018-01-31"
    assert trimmed["meta"]["n_obs"] == len(trimmed["close"]["dates"])
    assert len(fake_yahoo.calls) == calls


def test_series_missing_values_are_null_not_nan(client):
    content = "date,close\n2020-01-01,1\n2020-01-02,\n2020-01-03,1.5\n2020-01-06,#N/A\n"
    sid = upload(client, content).json()["series"]["snapshot_id"]
    r = client.get("/api/data/series", params={"snapshot_id": sid})
    assert r.status_code == 200
    assert "NaN" not in r.text
    close = r.json()["close"]
    assert close["dates"] == ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-06"]
    assert close["values"] == [1.0, None, 1.5, None]


@pytest.mark.parametrize("sid", ["no_existe_1d_abc", "../../etc/passwd"])
def test_series_unknown_snapshot_is_404(client, sid):
    r = client.get("/api/data/series", params={"snapshot_id": sid})
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "SNAPSHOT_NO_ENCONTRADO"


def test_series_empty_range_is_422(client):
    sid = load(client, ["KO"]).json()["series"][0]["snapshot_id"]
    r = client.get(
        "/api/data/series", params={"snapshot_id": sid, "start": "1990-01-01", "end": "1990-12-31"}
    )
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "SIN_DATOS_EN_RANGO"
    r = client.get(
        "/api/data/series", params={"snapshot_id": sid, "start": "2020-01-01", "end": "2019-01-01"}
    )
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "RANGO_FECHAS_INVALIDO"


# --------------------------------------------------------------------------- /data/validate


def ref(symbol, snapshot_id=None, provider="yahoo"):
    out = {"provider": provider, "symbol": symbol}
    if snapshot_id:
        out["snapshot_id"] = snapshot_id
    else:
        out.update(RANGE)
    return out


def test_validate_pair_ok(client):
    r = client.post("/api/data/validate", json={"a": ref("KO"), "b": ref("PEP")})
    assert r.status_code == 200
    body = r.json()
    assert body["blocked"] is False
    assert body["calendar"] == "bolsa"
    assert body["n_aligned"] > 1000
    assert body["synthetic"] is False
    assert body["disclaimer"] == DISCLAIMER
    assert [m["asset"]["symbol"] for m in body["series_meta"]] == ["KO", "PEP"]
    assert body["status_message"].startswith("Datos aptos")


def test_validate_with_snapshot_uses_only_cache(client, fake_yahoo: FakeYahoo):
    meta = load(client, ["KO", "PEP"]).json()["series"]
    calls = len(fake_yahoo.calls)
    r = client.post(
        "/api/data/validate",
        json={
            "a": ref("KO", meta[0]["snapshot_id"]),
            "b": ref("PEP", meta[1]["snapshot_id"]),
        },
    )
    assert r.status_code == 200
    assert len(fake_yahoo.calls) == calls
    assert [m["snapshot_id"] for m in r.json()["series_meta"]] == [
        meta[0]["snapshot_id"],
        meta[1]["snapshot_id"],
    ]


def test_validate_currency_mismatch_is_blocked_report(client):
    r = client.post("/api/data/validate", json={"a": ref("KO"), "b": ref("SAP.DE")})
    assert r.status_code == 200
    body = r.json()
    assert body["blocked"] is True
    assert body["currency_mismatch"] is True
    issue = next(i for i in body["issues"] if i["code"] == "MONEDA_DISTINTA")
    assert issue["severity"] == "error"
    assert "EUR" in issue["message"]
    assert body["status_message"].startswith("Análisis bloqueado")


def test_validate_insufficient_data(client):
    short = {"provider": "yahoo", "start": "2020-01-01", "end": "2020-06-30"}
    r = client.post(
        "/api/data/validate",
        json={"a": {**short, "symbol": "KO"}, "b": {**short, "symbol": "PEP"}},
    )
    body = r.json()
    assert body["blocked"] is True
    assert body["insufficient_data"] is True


def test_validate_synthetic_pair_is_flagged(client):
    syn = client.post("/api/data/synthetic", json={"seed": 4}).json()
    r = client.post(
        "/api/data/validate",
        json={
            "a": ref(syn["a"]["asset"]["symbol"], syn["a"]["snapshot_id"], "synthetic"),
            "b": ref(syn["b"]["asset"]["symbol"], syn["b"]["snapshot_id"], "synthetic"),
        },
    )
    assert r.status_code == 200
    assert r.json()["synthetic"] is True


def test_validate_snapshot_of_other_symbol_is_rejected(client):
    meta = load(client, ["KO"]).json()["series"]
    r = client.post(
        "/api/data/validate",
        json={"a": ref("PEP", meta[0]["snapshot_id"]), "b": ref("KO")},
    )
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["code"] == "REFERENCIA_INCONSISTENTE"
    assert "KO" in err["message"] and "PEP" in err["message"]


@pytest.mark.parametrize(
    ("body", "field"),
    [
        ({"a": {"provider": "yahoo", "symbol": "KO"}, "b": ref("PEP")}, "a"),
        (
            {"a": {**ref("KO"), "start": "2021-01-01", "end": "2020-01-01"}, "b": ref("PEP")},
            "a",
        ),
        ({"a": {**ref("KO"), "start": "2020-13-01"}, "b": ref("PEP")}, "a.start"),
        ({"a": ref("KO"), "b": ref("PEP"), "spread_type": "lineal"}, "spread_type"),
        ({"a": ref("KO"), "b": ref("PEP"), "train_fraction": 0}, "train_fraction"),
        ({"a": ref("KO")}, "b"),
    ],
)
def test_validate_request_errors(client, body, field):
    r = client.post("/api/data/validate", json=body)
    assert r.status_code == 422
    assert field in [c["campo"] for c in r.json()["error"]["details"]["campos"]]


def test_validate_provider_failure_status(client):
    r = client.post("/api/data/validate", json={"a": ref("FALLA_RED"), "b": ref("PEP")})
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "ERROR_RED"


def test_concurrent_loads_share_the_cache_safely(client, fake_yahoo: FakeYahoo):
    """Peticiones simultáneas (hilos del servidor): ninguna instantánea se pierde del índice."""
    symbols = ["KO", "PEP", "XOM", "CVX", "SPY", "IVV", "^GSPC", "SAP.DE"]
    results: dict[str, int] = {}
    barrier = threading.Barrier(len(symbols))

    def work(sym: str) -> None:
        barrier.wait()
        results[sym] = load(client, [sym]).status_code

    threads = [threading.Thread(target=work, args=(s,)) for s in symbols]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    assert results == dict.fromkeys(symbols, 200)
    cached = client.get("/api/data/cache", params={"provider": "yahoo"}).json()
    assert {m["asset"]["symbol"] for m in cached} == set(symbols)
    calls = len(fake_yahoo.calls)
    again = load(client, symbols).json()
    assert len(fake_yahoo.calls) == calls  # todo servido desde la caché
    assert again["cached"] == symbols
