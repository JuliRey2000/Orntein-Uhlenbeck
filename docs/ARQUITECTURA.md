# Arquitectura — Laboratorio de Reversión a la Media

> Herramienta educativa. No constituye asesoría de inversión. Resultados
> históricos y simulados no garantizan resultados futuros.

Plan aprobado el 2026-10-10. Las decisiones de detalle están en
[`DECISIONS.md`](DECISIONS.md).

## 1. Estructura de carpetas

```
.
├── backend/
│   ├── pyproject.toml              # dependencias, ruff, pytest
│   ├── providers.toml              # registro de proveedores de datos
│   ├── src/mrlab/
│   │   ├── config.py               # configuración desde .env
│   │   ├── domain/                 # modelos compartidos y errores (mensajes en español)
│   │   ├── data/                   # proveedores, registro, caché, servicio, validación, calendario, proxies, presets
│   │   ├── stats/                  # ADF/KPSS, Engle–Granger + estabilidad, BH, correlaciones, ventanas
│   │   ├── model/                  # spread (MCO), OU (estimación, pronóstico, simulación exacta), diagnóstico
│   │   ├── strategy/               # parámetros, máquina de estados, dimensionamiento
│   │   ├── backtest/               # división de la muestra, motor, costes, métricas
│   │   ├── analysis/               # orquestación de un par (estado + causas), ejecución desde RunConfig
│   │   ├── reporting/              # run_config (hash, git), exportación CSV/JSON/ZIP (HTML en fase 3)
│   │   ├── studies.py              # estudios: registro persistente de la prueba (fase 2)
│   │   └── api/                    # FastAPI (fase 2): routers, DTO, errores, trabajos, OpenAPI
│   ├── scripts/export_openapi.py   # backend/openapi.json (make openapi)
│   ├── Dockerfile
│   └── tests/{unit,integration,api}/
├── docker-compose.yml              # servicio backend (frontend en la fase 3)
├── frontend/                       # React + TS + Vite (fase 3)
├── data/cache/  data/studies/  data/uploads/   # locales, ignorados por git
├── docs/                           # ARQUITECTURA, MATEMATICAS, DECISIONS, GUIA_ALUMNO
└── examples/                       # CSV SINTÉTICOS + demo_ko_pep.py
```

El núcleo (`mrlab` sin `api/`) es una librería pura y determinista: sin estado
global; solo los proveedores usan la red. La API y el frontend son capas finas.

## 2. Flujo de análisis de un par

```
cargar (caché/proveedor) → validar → alinear calendarios → dividir 60/20/20 (fechas congeladas)
→ ADF/KPSS por serie → Engle–Granger en ambas direcciones → BH → estabilidad por ventanas
→ MCO α, β (solo formación) → spread → AR(1) exacto → OU (o «no estacionario») → diagnóstico
→ estado + causas + «¿Por qué se descartó?» → estrategia → backtest → métricas → exportación
```

Un único `RunConfig` (pydantic) parametriza todo; se serializa como
`run_config.json` y su hash SHA-256 (JSON canónico) identifica la ejecución.

### Dependencias entre módulos

```
domain ← data ← stats ← model ← strategy ← backtest ← analysis ← reporting ← studies ← api
```

(cada capa solo importa de capas a su izquierda; `data.providers.synthetic` usa
`model.ou.simulate_ou_exact`).

## 3. Contratos de la API (fase 2)

Prefijo `/api`. Las peticiones de análisis llevan el `RunConfig` completo (sin
estado y deterministas) y la respuesta devuelve la configuración **fijada**
(instantáneas y fechas de corte); el único estado de la API es el **estudio**
(registro persistente de la prueba, en `data/studies/`) y la lista de trabajos en
segundo plano (en memoria). El contrato completo, con cada campo descrito en
español, está en `/docs` y en `backend/openapi.json` (versionado; `make openapi`
lo regenera y una prueba comprueba que está al día).

| Método | Ruta | Entrada → salida |
|---|---|---|
| GET | `/health` | → `HealthOut` {status, version, code_version (hash git), disclaimer} |
| GET | `/providers` | → `ProviderOut[]` {name, label, description, kind, is_default, synthetic, notices} (sin claves) |
| GET | `/examples` | → `ExampleOut[]` (pares precargados; ninguno se asume cointegrado) |
| GET | `/proxies` | → `ProxySuggestionOut[]` (índice/futuro → proxy sugerido) |
| GET | `/glossary` | → `GlossaryOut` (textos para tooltips) |
| GET | `/assets/search?q=&provider=` | → `AssetInfoOut[]` |
| GET | `/assets/metadata?provider=&symbol=` y `/assets/{provider}/{symbol}` | → `AssetMetadataOut` {asset, requires_proxy, proxy (sugerido), synthetic, note} |
| POST | `/data/load` | `{provider, symbols (1..30), start, end, refresh}` → `LoadResponse` {series: `SeriesMeta[]` con `snapshot_id`, errors por símbolo, cached, notices} |
| POST | `/data/csv` | multipart: archivo + symbol, currency, asset_class, price_type (origin, seed) → `CSVUploadOut` (413 si supera `MRLAB_MAX_UPLOAD_MB`) |
| POST | `/data/synthetic` | escenario + parámetros + semilla → `SyntheticOut` {a, b: `SeriesMeta`, label «SINTÉTICO», params} |
| GET | `/data/cache?provider=&symbol=` | → `SeriesMeta[]` de la caché (nunca descarga) |
| GET | `/data/series?snapshot_id=&start=&end=` | → `SeriesDataOut` {meta, close} (solo caché; nunca rellena fechas) |
| POST | `/data/validate` | `{a, b: SeriesQueryIn, spread_type, min_train_obs, train_fraction}` → `ValidationOut` (`ValidationReport` + series) |
| POST | `/correlations` | `{series (2..30), window, pair}` → `CorrelationOut` (Pearson/Spearman + n, base 100, correlación móvil, rendimientos) |
| POST | `/pairs/analyze` | `RunConfig` → `AnalysisOut` {config fijado, analysis: `PairAnalysisOut`} |
| POST | `/ou/simulate` | λ, μ, σ, X₀, horizonte (≤ 2.520), trayectorias (≤ 500), semilla, paths_returned → `SimulationOut` |
| POST | `/backtest` | `RunConfig` → `BacktestOut` (segmentos formación y validación; nunca la prueba) |
| POST | `/backtest/compare` | `{configs: RunConfig[] (2..10)}` → 202 `JobAccepted`; resultado `CompareResult` en `/jobs/{id}` |
| GET / DELETE | `/jobs`, `/jobs/{id}` | → `JobSummaryOut[]`, `JobOut` (progreso, resultado o error); DELETE cancela (409 si ya terminó) |
| POST | `/studies` | `{config: RunConfig, name}` → 201 `StudyOut` (congela instantáneas y división) |
| GET | `/studies`, `/studies/{id}` | → `StudySummaryOut[]`, `StudyOut` {study, test_viewed, contaminated, contamination_reason, contamination_text} |
| PUT | `/studies/{id}/config` | `{strategy?, costs?, analysis?}` (cambios parciales) → `StudyOut` (+ `warning` si contamina) |
| POST | `/studies/{id}/test` | → `StudyTestOut` {study, backtest con «prueba», contaminated, contamination_text} |
| GET | `/studies/{id}/export?excel_es=` | → ZIP del estudio (con la marca de contaminación del registro) |
| DELETE | `/studies/{id}` | → 204 (el registro de una prueba vista pasa a `data/studies/eliminados/`) |
| POST | `/export` | `{config: RunConfig, excel_es}` → ZIP (prefijo `SINTETICO_` si aplica) |
| POST | `/runs/replay`, `/runs/replay/json` | multipart `run_config.json` → mismo ZIP byte a byte / `ReplayOut` (solo caché) |
| POST | `/universe/rank` → `job_id` | ranking con progreso (fase 4) |
| POST | `/walkforward`, `/portfolio` | fases 4 y 5 |

Errores: `{"error": {"code": "...", "message": "<español>", "details": {...}}}`
(`api/errors.py`). Validación de la petición → 422 `PETICION_INVALIDA` con
`details.campos = [{campo, motivo, recibido}]` (nunca el texto de pydantic en
inglés). 404 recurso inexistente; 409 trabajo ya terminado; 413 archivo demasiado
grande; 429 límite del proveedor (con `Retry-After`); 502 fallo del proveedor; 503
sin conexión. Las rutas locales del servidor (caché, estudios) nunca llegan al
cliente. Los tipos TypeScript se generan de `backend/openapi.json`
(`openapi-typescript`, fase 3).

Despliegue: un solo proceso de uvicorn (decisión F1). `backend/Dockerfile` (Python
3.12, usuario sin privilegios, `HEALTHCHECK` a `/api/health`) y
`docker-compose.yml` (servicio `backend`, volumen `./data → /app/data`, puerto
8000; el servicio `frontend` entra en la fase 3).

## 4. Frontend (fase 3)

React + TypeScript + Vite + Tailwind; `react-plotly.js` con `plotly.js-dist-min`
empaquetado (funciona sin internet); Zustand; KaTeX para «Aprende».
Componentes transversales: `AvisoEducativo` (todas las pestañas),
`EtiquetaSintetico`, `EstadoPar`, `PanelDescarte`, `FichaSerie`, `InfoTooltip`.
Formato numérico `10.000`, `4.214`, `0,05` (agrupación forzada; `Intl` es-ES no
agrupa números de 4 cifras).

## 5. Caché y reproducibilidad

- Instantáneas Parquet inmutables, direccionadas por contenido:
  `data/cache/<proveedor>/<símbolo>/<snapshot_id>.parquet` + `index.json`.
- `run_config.json` referencia los `snapshot_id`; reproducir nunca re-descarga en
  silencio.
- Exportación byte a byte idéntica (ver decisión I4).

## 6. Fases

| Fase | Contenido | Criterio de terminado |
|---|---|---|
| 1 | Núcleo, exportación CSV/JSON, `run_config`, `demo_ko_pep.py` | Pruebas de la sección 14; demo desde CSV |
| 2 | API, trabajos, errores en español | `/docs`; pruebas de integración |
| 3 | 9 pestañas, informe HTML | Flujo completo; modo sintético sin internet |
| 4 | Ranking FDR, walk-forward, recalibración de β, umbrales en validación | Pruebas de FDR y walk-forward |
| 5 | Cartera multi-par | Consolidado = suma de pares |
