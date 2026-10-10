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
│   │   └── api/                    # FastAPI (fase 2)
│   └── tests/{unit,integration}/
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
domain ← data ← stats ← model ← strategy ← backtest ← analysis ← reporting ← api
```

(cada capa solo importa de capas a su izquierda; `data.providers.synthetic` usa
`model.ou.simulate_ou_exact`).

## 3. Contratos de la API (fase 2)

Prefijo `/api`. Las peticiones de análisis llevan el `RunConfig` completo (sin
estado y deterministas); el único estado es el «estudio» para detectar prueba
contaminada.

| Método | Ruta | Entrada → salida |
|---|---|---|
| GET | `/health` | → versión, hash git |
| GET | `/providers` | → proveedores y avisos (Yahoo no oficial) |
| GET | `/assets/search?q=&provider=` | → `AssetInfo[]` |
| GET | `/assets/{provider}/{symbol}` | → `AssetInfo` (ejecutable, proxy sugerido) |
| GET | `/examples` | → pares precargados |
| POST | `/data/load` | `{provider, symbols, start, end}` → `SeriesMeta[]` con `snapshot_id` |
| POST | `/data/csv` | archivo + moneda, tipo de precio, clase de activo → `SeriesMeta` |
| POST | `/data/synthetic` | escenario + parámetros + semilla → `SeriesMeta[]` |
| POST | `/data/validate` | `SeriesRef[]` → `ValidationReport` |
| POST | `/correlations` | series, ventana → Pearson/Spearman + n, correlación móvil, base 100 |
| POST | `/pairs/analyze` | `RunConfig` → `PairAnalysis` |
| POST | `/ou/simulate` | λ, μ, σ, X₀, horizonte, trayectorias ≤ 500, semilla → simulación |
| POST | `/backtest` | `RunConfig` + segmento → `BacktestResult` |
| POST | `/studies`, `/studies/{id}/test` | congela configuración; prueba una vez; marca contaminación |
| POST | `/backtest/compare` | varios pares, cada uno con su capital |
| POST | `/universe/rank` → `job_id`; GET `/jobs/{id}` | ranking con progreso (fase 4) |
| POST | `/walkforward`, `/portfolio` | fases 4 y 5 |
| POST | `/export` | → ZIP (CSV, JSON, HTML) |
| POST | `/runs/replay` | `run_config.json` → mismo análisis desde la caché |

Errores: `{"error": {"code": "...", "message": "<español>", "details": {...}}}`.
Los tipos TypeScript se generan del OpenAPI (`openapi-typescript`).

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
