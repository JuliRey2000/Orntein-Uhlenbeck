# Especificación — Laboratorio de Reversión a la Media

Documento de requisitos original (2026-10-10). Las decisiones que lo concretan o
lo interpretan están en [`DECISIONS.md`](DECISIONS.md).

## 0. Rol, contexto y forma de trabajar

Aplicación web **educativa** para investigar y simular pairs trading
multi-activo, usada primero por el responsable del proyecto y luego compartida
con los alumnos de su mentoría de trading para explicar el fenómeno de la
reversión a la media. Por eso:

- **Toda la interfaz, textos de ayuda, mensajes de error y documentación van en
  español.** El código (nombres de variables, funciones, commits) puede ir en inglés.
- Las explicaciones deben ser claras para alguien que sabe operar pero no domina
  econometría: intuición primero, fórmula después, y un ejemplo numérico cuando ayude.
- La herramienta **nunca** debe sugerir que un par "funciona" sin evidencia. Su
  valor pedagógico está precisamente en mostrar cuándo la estrategia **no** aplica
  y por qué.
- Debe mostrar en todas las pestañas un aviso visible: *"Herramienta educativa.
  No constituye asesoría de inversión. Resultados históricos y simulados no
  garantizan resultados futuros."*

### Forma de trabajar (obligatoria)

1. Antes de escribir código: plan de arquitectura, supuestos y decisiones;
   esperar aprobación.
2. Trabajar **por fases** (sección 13). No avanzar a la siguiente fase hasta que
   todas las pruebas de la fase actual pasen.
3. Toda decisión no especificada se registra en `docs/DECISIONS.md` con:
   decisión, alternativas consideradas, justificación y fecha.
4. Al terminar cada fase: resumen corto (qué se hizo, cómo probarlo, qué quedó pendiente).

## 1. Entorno y stack

- **Backend:** Python 3.11+, FastAPI, pandas, NumPy, SciPy, statsmodels, pydantic v2.
- **Frontend:** React + TypeScript + Vite, Plotly (`react-plotly.js`), UI ligera (Tailwind).
- **Pruebas:** pytest (backend), Vitest (frontend, al menos utilidades de formato y estado).
- **Arranque:** `docker compose up` levanta todo. `Makefile` con `make install`,
  `make dev`, `make test`, `make demo`.
- **Caché:** Parquet locales en `data/cache/`, con metadatos (proveedor, símbolo,
  fecha de descarga, rango cubierto, tipo de precio).
- **Configuración:** `.env` (con `.env.example`). Ninguna clave de API llega al navegador.
- Debe funcionar en Windows, macOS y Linux.

## 2. Datos

### 2.1 Proveedor intercambiable

```python
class DataProvider(Protocol):
    name: str
    def search(self, query: str) -> list[AssetInfo]: ...
    def get_history(self, symbol: str, start: date, end: date,
                    interval: str = "1d") -> PriceSeries: ...
    def get_metadata(self, symbol: str) -> AssetInfo: ...
```

1. **`YahooProvider`** (por defecto) con `yfinance`.
   - Precios **ajustados por splits y dividendos** (`auto_adjust=True`); **no**
     sumar dividendos por separado en el PnL (documentar para evitar doble conteo).
   - API no oficial: puede fallar, limitar peticiones o cambiar sin aviso.
     Reintentos con espera exponencial, caché agresiva, errores claros en español.
   - Mostrar en la interfaz que Yahoo Finance es una fuente gratuita no oficial,
     adecuada para uso educativo.
2. **`CSVProvider`**: CSV con columnas `date, close` (opcional `open, high, low,
   volume`), con validación y mensajes de error claros.
3. **`SyntheticProvider`**: pares sintéticos (cointegrados y no cointegrados) con
   semilla fija. Siempre etiquetado como **SINTÉTICO** en toda la interfaz y en
   las exportaciones.

Debe ser trivial añadir otro proveedor (Tiingo, Polygon, Binance) implementando
la interfaz y registrándolo en un archivo de configuración.

### 2.2 Metadatos que se muestran de cada serie

| Campo | Ejemplo |
|---|---|
| Nombre económico | S&P 500 |
| Símbolo de la serie analizada | `^GSPC` (Yahoo) |
| Instrumento ejecutable para simular | `SPY` |
| Moneda | USD |
| Tipo de precio | Cierre ajustado |
| Proveedor | Yahoo Finance |
| Cobertura | 2010-01-04 a 2026-10-09, 4.214 observaciones |
| Fecha de descarga | 2026-10-10 11:30 |

**Índices no ejecutables:** si el usuario elige un índice, la app le pide elegir
un proxy ejecutable y sugiere uno, sin sustituirlo en silencio.

| Índice / referencia | Símbolo Yahoo | Proxy sugerido |
|---|---|---|
| S&P 500 | `^GSPC` | `SPY` |
| Nasdaq 100 | `^NDX` | `QQQ` |
| Oro | `GC=F` | `GLD` |
| Bitcoin | `BTC-USD` | `BTC-USD` (spot, con nota sobre cortos) |

Pares de ejemplo precargados: KO/PEP, SPY/GLD, QQQ/BTC-USD, XOM/CVX, y un par
sintético cointegrado. **No asumir que ninguno está cointegrado.**

### 2.3 Validación de datos

Antes de cualquier análisis, ejecutar y mostrar un informe de validación:

- Orden temporal y fechas duplicadas (se eliminan duplicados conservando el
  último, con aviso).
- Valores faltantes: se reportan, **no** se rellenan con forward-fill para crear
  observaciones nuevas.
- Precios estrictamente positivos (requisito para logaritmos). Si no, bloquear el
  spread logarítmico.
- Historial común mínimo: **500 observaciones** en la muestra de formación. Si no
  se cumple, estado "Datos insuficientes".
- Moneda: si los dos activos tienen monedas distintas, bloquear el análisis en la
  v1 y explicar por qué (la conversión de moneda queda como extensión).
- Saltos sospechosos (rendimiento diario > 50 %) se marcan como advertencia por
  posible error de ajuste.

### 2.4 Sincronización de calendarios

- **Dos activos de bolsa:** intersección de fechas de negociación.
- **Dos criptomonedas:** calendario de 7 días.
- **Cripto + bolsa:** solo los días de negociación de la bolsa (intersección).
  Nunca rellenar fines de semana ni festivos. Advertir del desfase: en Yahoo el
  cierre diario de cripto es a medianoche UTC; la bolsa de EE. UU. cierra a las
  16:00 de Nueva York.

### 2.5 Frecuencia

Solo datos **diarios** en la v1. La frecuencia es un parámetro (Δt), no un número
fijo escrito a mano.

## 3. Explorador de relaciones (pestaña "Correlaciones")

- Matriz de correlación de **rendimientos logarítmicos** (Pearson y Spearman), con
  número de observaciones por celda.
- Correlación móvil con ventana configurable (default 63 días).
- Precios normalizados (base 100), rendimientos y dispersión.
- Tres conceptos separados visualmente, con explicación corta: correlación,
  cointegración, aptitud para OU.
- Ejemplo pedagógico: muy correlacionados y no cointegrados (y al revés), con el
  generador sintético.
- La correlación es un **filtro exploratorio opcional** (default: desactivado),
  nunca una prueba de cointegración.

## 4. Cointegración (pestaña "Cointegración y ranking")

### 4.1 Integración de cada serie
- **ADF** (rezagos por AIC, máximo 12, con constante) y **KPSS** (con constante)
  sobre log-precios y primeras diferencias.
- Interpretación combinada: "compatible con I(1)", "posiblemente estacionaria",
  "resultado ambiguo". Si alguna serie no parece I(1): "Supuestos dudosos".

### 4.2 Engle–Granger
- `statsmodels.tsa.stattools.coint` (p-values de MacKinnon para residuos
  estimados), `trend="c"`, rezagos por AIC.
- **Prohibido** usar el p-value de un ADF ordinario sobre los residuos como si
  fuera el de Engle–Granger. Escribir una prueba que lo verifique.
- Ambas direcciones; por defecto la de menor p-value, indicándolo.
- Mostrar: estadístico, p-value, valores críticos (1 %, 5 %, 10 %), nivel elegido
  (default 5 %), ventana, observaciones, tendencia, rezagos.

### 4.3 Pruebas múltiples
Benjamini–Hochberg (FDR, q = 0,05) sobre todos los pares y direcciones. p-value
original y ajustado lado a lado, con explicación.

### 4.4 Estabilidad entre ventanas
Ventanas de 252 días, paso 63, dentro de la formación. "Resultado inestable" si en
más del 40 % de las ventanas el p-value supera el umbral, o si β cambia de signo.
Gráfico de p-value y β por ventana.

### 4.5 Estados posibles de un par (exactamente estos, con causa)
- ✅ Evidencia de cointegración en la ventana evaluada
- ⚠️ Evidencia insuficiente
- ⚠️ Supuestos dudosos
- ⚠️ Resultado inestable entre ventanas
- ⛔ Datos insuficientes

Nunca "cointegración garantizada" o "par rentable".

## 5. Construcción del spread

```
Precios:      X_t = P_A,t − α − β · P_B,t
Logarítmico:  X_t = log(P_A,t) − α − β · log(P_B,t)      (default)
```

- α y β por **MCO solo con formación**, **fijos** en validación y prueba (v1).
- Mostrar α, β, error estándar, R², método, periodo, tipo de spread, unidades e
  interpretación (log: "β ≈ elasticidad: un 1 % en B se asocia con un β % en A";
  precios: "β unidades de B por cada unidad de A").
- **Consistencia α–μ:** el residuo tiene media ≈ 0 en formación; el μ del OU se
  estima sobre ese residuo y se usa tal cual. No restar dos veces el mismo nivel.
  Escribir una prueba.
- **Extensión (fase 4):** recalibración móvil de β, con operaciones y costes de
  rebalanceo, y explicación de que un spread redefinido continuamente no es un
  único OU de parámetros constantes.

## 6. Proceso Ornstein–Uhlenbeck (pestaña "Spread y modelo OU")

```
dX_t = λ(μ − X_t) dt + σ dW_t      (λ ≡ θ en otras notaciones)
```

### 6.1 Estimación exacta discreta (Δt = 1 día de negociación)

```
X_{t+Δt} = a + b · X_t + ε_t,    ε_t ~ N(0, s²)
Si 0 < b < 1:
  λ = −ln(b)/Δt ;  μ = a/(1 − b) ;  σ = sqrt(2λ·s²/(1 − b²))
  σ_eq = σ/sqrt(2λ) ;  vida media = ln(2)/λ
E[X_{t+h} | X_t] = μ + (X_t − μ)·e^{−λh}
Var[X_{t+h} | X_t] = σ_eq²·(1 − e^{−2λh})
```

- Si **b ≥ 1** o **b ≤ 0**: advertencia clara ("el spread no muestra reversión
  estacionaria compatible con un OU") y **no** mostrar vida media, λ ni bandas.
- Vida media > 126 días: demasiado lenta para operar con este horizonte.
- Vida media < 1 día: con datos diarios no es operable ni fiable.
- Unidades visibles: λ en 1/día de negociación, vida media en días de
  negociación, σ en unidades del spread por raíz de día.
- `z_t = (X_t − μ)/σ_eq` con μ y σ_eq **fijos de formación** (z móvil opcional,
  solo como comparación pedagógica).

### 6.2 Diagnóstico
- Spread con μ y bandas ±1, ±2 σ_eq; pronóstico a h = 1, 5, 20 con intervalos.
- Residuos AR(1), ACF (20 rezagos), Ljung–Box.
- Estabilidad de λ, μ, σ por ventanas (las de 4.4).
- Distinguir: bandas estacionarias; intervalos predictivos condicionales;
  incertidumbre de parámetros (no incluida en la v1, avisarlo; bootstrap por
  bloques como extensión).
- μ es la **"referencia estadística del modelo"**, nunca "valor justo".

### 6.3 Modo educativo / sintético (pestaña "Simulación sintética")
- Controles: λ, μ, σ, X₀, horizonte, trayectorias (máx. 500), semilla.
- Simulación por la **transición exacta** (no Euler).
- Trayectorias, media y bandas teóricas, histograma del tiempo hasta cruzar μ,
  vida media teórica vs. empírica.
- Botón "Generar un par sintético cointegrado con estos parámetros".
- Visualmente distinto, etiqueta **SINTÉTICO** permanente; nunca modifica datos
  históricos.

## 7. Estrategia (pestaña "Estrategia")

### 7.1 Parámetros por defecto

| Parámetro | Default |
|---|---|
| Entrada (z) | ±2,0 |
| Salida (z) | ±0,5 |
| Stop-loss (z) | ±3,5 |
| Stop-loss monetario | 5 % del capital asignado al par |
| Tiempo máximo | 3 × vida media (redondeado, mínimo 5 días) |
| Enfriamiento tras stop | 5 días de negociación |
| Capital inicial | 10.000 USD |
| Exposición bruta objetivo | 100 % del capital |
| Apalancamiento máximo | 2× |
| Ventas en corto | Permitidas (configurable) |

### 7.2 Reglas
- z > +entrada → **vender el spread** (corto A, largo β·B).
- z < −entrada → **comprar el spread** (largo A, corto β·B).
- Salida cuando |z| ≤ salida, por stop (z o monetario) o por tiempo máximo.
- Si β < 0, ambas piernas van en la misma dirección (manejarlo y explicarlo).

### 7.3 Dimensionamiento
Spread logarítmico (β relaciona nocionales):
```
q_A = ± N / P_A ;  q_B = ∓ β · N / P_B
Exposición bruta = N · (1 + |β|)  →  N = Exposición_bruta_objetivo / (1 + |β|)
```
Spread en precios: `q_B = ∓ β · q_A`. Fracciones de unidad por defecto, opción de
redondear. Respetar el apalancamiento máximo (reducir y registrar la razón).

### 7.4 Sin ventas en corto
Las señales que requieren cortar una pierna se registran pero no se ejecutan
("señal no ejecutable"). Explicar que la estrategia de pares requiere ambas
piernas; en cripto spot el corto requiere derivados (fuera de la v1).

### 7.5 Máquina de estados
`SIN_POSICION`, `LARGO_SPREAD`, `CORTO_SPREAD`, `CERRANDO`, `ENFRIAMIENTO`.
- Sin entradas con posición abierta o en enfriamiento.
- Prioridad: stop monetario > stop z > tiempo máximo > salida normal. Registrar
  todas las condiciones activas y la aplicada.
- Todo a precio de cierre según la convención de 8.2.

## 8. Backtest sin información futura

### 8.1 División
Cronológica 60/20/20. Selección, α, β, OU: solo formación. Ajuste de umbrales:
solo validación. Prueba una vez con configuración congelada; si se cambian
umbrales después, "prueba contaminada".

### 8.2 Convención de ejecución
Señal con el cierre de t (solo información hasta t); ejecución al cierre de t+1.

### 8.3 Modos
Un par; comparación entre pares (cada uno con su capital); walk-forward (fase 4:
formación 504, prueba 126, paso 126).

## 9. PnL y costes

```
ΔΠ_t = q_A,t−1 · ΔP_A,t + q_B,t−1 · ΔP_B,t − C_t
```

**Nunca** calcular el PnL monetario como la diferencia del spread logarítmico.

| Coste | Default |
|---|---|
| Comisión por pierna | 5 pb del nocional operado |
| Bid-ask / slippage | 5 pb por pierna (acciones y ETF), 10 pb (cripto) |
| Préstamo de cortos | 3 % anual sobre el nocional corto, devengado diario |
| Financiación del apalancamiento | 5 % anual sobre el exceso sobre 1× |
| Rebalanceo | Mismos costes en cada ajuste de cantidades |

Dividendos ya incluidos en el precio ajustado. Mostrar por separado: PnL bruto,
costes por tipo, PnL neto, realizado y no realizado, capital, exposición bruta y
neta, apalancamiento.

## 10. Métricas (cada una con tooltip de una línea)

Retorno neto total y anualizado; máximo drawdown y duración; Sharpe (rendimientos
diarios netos, rf = 0 declarada, 252 para acciones/ETF/mixtos, 365 solo cripto,
convención junto al número); número de operaciones, % aciertos, ganancia media,
pérdida media, factor de beneficio; duración media; rotación; exposición media y
apalancamiento máximo; costes totales y % del PnL bruto. Con menos de 10
operaciones: "muestra insuficiente para conclusiones estadísticas".

## 11. Interfaz (pestañas)

1. Universo y datos · 2. Correlaciones · 3. Cointegración y ranking · 4. Spread y
modelo OU · 5. Estrategia · 6. Backtest y riesgo · 7. Simulación sintética ·
8. Exportación · 9. Aprende.

Gráficos interactivos: precios normalizados, correlación móvil, spread con
bandas, z-score con umbrales, entradas y salidas, curva de capital neta,
drawdown, histograma de resultados por operación, trayectorias simuladas.
Panel lateral "¿Por qué se descartó este par?" con razones concretas.

## 12. Ranking de pares

Puntaje compuesto con pesos visibles: evidencia de cointegración (p ajustado),
estabilidad, vida media (5 a 63 días ideal), calidad del OU (Ljung–Box,
estabilidad de λ), costes relativos a σ_eq, resultado en **validación** (nunca en
prueba). No ordenar por PnL retrospectivo; la prueba va en columna aparte.
Universo v1: hasta 30 activos (435 pares, 870 direcciones), en segundo plano con
barra de progreso. Cartera multi-par en fase 5.

## 13. Fases

| Fase | Contenido | Criterio de terminado |
|---|---|---|
| 1 | Núcleo Python: proveedores, validación, cointegración, spread, OU, backtest de un par, métricas | Pruebas de la sección 14; `examples/demo_ko_pep.py` reproduce un análisis completo desde CSV |
| 2 | API FastAPI, caché, errores en español | `/docs`; pruebas de integración |
| 3 | Frontend 9 pestañas y exportación | Flujo completo; demo sintético sin internet |
| 4 | Ranking FDR, estabilidad, walk-forward, recalibración de β | Pruebas de FDR y walk-forward |
| 5 | Cartera multi-par | Consolidado = suma de pares |

## 14. Pruebas obligatorias

- **Recuperación OU:** 5.000 pasos, λ = 0,05, μ = 0, σ = 0,02, semilla 42; λ con
  error relativo < 15 %, σ < 10 %, μ dentro de 3 errores estándar (decisión O2).
- **b fuera de rango:** random walk → "no estacionario" y sin vida media.
- **Cointegración sintética:** par cointegrado detectado en ≥ 90 % de 100
  semillas; dos random walks independientes rechazados en ≥ 90 %.
- **Coincidencia con statsmodels:** p-value de EG = `coint` y ≠ ADF ordinario
  sobre residuos.
- **FDR:** igual a `multipletests(method="fdr_bh")`.
- **Anti look-ahead:** modificar precios posteriores a t no cambia señales,
  posiciones ni PnL hasta t.
- **Consistencia α–μ:** μ estimado sobre el residuo de formación ≈ 0.
- **Contabilidad:** bruto − costes = neto; capital final = inicial + Σ neto;
  realizado + no realizado = total.
- **β negativo:** ambas piernas en la misma dirección y PnL correcto en un caso
  construido a mano.
- **Máquina de estados:** sin entradas duplicadas; prioridad de salidas respetada.
- **Validación de datos:** duplicados, faltantes, precios ≤ 0 y monedas distintas
  producen el estado y mensaje esperados.
- **Reproducibilidad:** misma configuración + mismos datos en caché = exportación
  idéntica byte a byte.

## 15. Reproducibilidad y exportación

`run_config.json` con activos, proveedor, fechas, versión del código (hash git),
fecha de descarga de cada serie, parámetros y semillas, hash de la configuración.
Cargarlo reproduce el análisis (con la caché). Exportar: operaciones (CSV), serie
diaria de posiciones/PnL/capital (CSV), pruebas estadísticas (CSV), configuración
(JSON) e informe HTML autocontenido. Toda exportación indica si los datos son
históricos o sintéticos.

## 16. Documentación

`README.md` (español), `docs/MATEMATICAS.md`, `docs/DECISIONS.md`,
`docs/GUIA_ALUMNO.md` (recorrido de 15 minutos), `examples/` con CSV marcado como
sintético y script reproducible.

## 17. Prohibiciones explícitas

- No presentar datos sintéticos como históricos.
- No rellenar días sin negociación y tratarlos como observaciones reales.
- No usar información posterior al momento de decisión.
- No optimizar umbrales sobre el periodo de prueba.
- No mostrar vida media ni bandas cuando el ajuste OU no es válido.
- No usar lenguaje promocional ("estrategia ganadora", "par rentable",
  "cointegración garantizada").
- No sustituir en silencio un índice por un proxy.
- No exponer claves de API al navegador.
