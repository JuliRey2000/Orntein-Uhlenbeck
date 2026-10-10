# Laboratorio de Reversión a la Media

> **Herramienta educativa. No constituye asesoría de inversión. Resultados
> históricos y simulados no garantizan resultados futuros.**

Laboratorio para investigar y simular **pairs trading** con dos herramientas
estadísticas: la **cointegración** (¿existe una combinación de dos precios que
vuelve a un nivel estable?) y el proceso de **Ornstein–Uhlenbeck** (¿a qué
velocidad vuelve?). Está pensado para los alumnos de una mentoría de trading: sabes
operar, pero no necesitas dominar la econometría.

Su valor pedagógico está en mostrar **cuándo la estrategia no aplica y por qué**.
El laboratorio nunca afirma que un par «funciona» sin evidencia, y explica cada
descarte con causas concretas.

## Contenido

- [Qué es y qué no es](#qué-es-y-qué-no-es)
- [Estado actual](#estado-actual)
- [Requisitos](#requisitos)
- [Instalación sin Docker](#instalación-sin-docker)
- [Pruebas](#pruebas)
- [Demo](#demo)
- [Fuentes de datos](#fuentes-de-datos)
- [Limitaciones de la v1](#limitaciones-de-la-v1)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Documentación](#documentación)

## Qué es y qué no es

**Es:**

- Un laboratorio para aprender, con datos históricos o sintéticos, a distinguir
  **correlación**, **cointegración** y **aptitud para un modelo OU**.
- Un flujo completo y auditable: validación de datos → pruebas ADF/KPSS →
  Engle–Granger en ambas direcciones con p-values de MacKinnon → corrección por
  pruebas múltiples (Benjamini–Hochberg) → estabilidad por ventanas → spread →
  ajuste OU → estrategia → backtest sin información futura, con costes →
  métricas.
- Reproducible: la misma configuración con los mismos datos en caché produce
  exactamente los mismos resultados.

**No es:**

- Asesoría de inversión ni un servicio de señales.
- Un sistema de trading: no se conecta a ningún broker ni envía órdenes reales.
- Un buscador de «pares que funcionan». Un estado ✅ significa «hay evidencia de
  cointegración en la ventana evaluada», no una garantía de nada.

Los datos sintéticos se etiquetan **SINTÉTICO** en todas partes (símbolos
`SINT-*`, nombres «SINTÉTICO — …», columna `tipo_datos` y prefijo
`SINTETICO_` en los CSV de ejemplo y en las exportaciones) y nunca se presentan
como históricos: aunque copies solo un CSV sintético y lo declares como histórico,
el laboratorio lo sigue tratando como SINTÉTICO y te lo avisa.

## Estado actual

El proyecto avanza por fases. **La fase 1, el núcleo de cálculo en Python, está
terminada** (paquete `mrlab`, en `backend/src/mrlab`), con sus pruebas y una demo
por línea de comandos. Las fases 2 y 3 son las siguientes.

| Fase | Contenido | Estado |
|---|---|---|
| 1 | Núcleo Python: proveedores de datos, validación, cointegración, spread, OU, backtest de un par, métricas, exportación CSV/JSON, demo | **Terminada** (dos rondas de revisión) |
| 2 | API FastAPI, caché, errores en español | Pendiente |
| 3 | Interfaz web (9 pestañas), informe HTML, Docker, `docs/GUIA_ALUMNO.md` | Pendiente |
| 4 | Ranking de pares con FDR, walk-forward, recalibración de β | Pendiente |
| 5 | Cartera multi-par | Pendiente |

**Estado de la fase 1.** El núcleo y la demo están completos y cubiertos por
pruebas automáticas que no necesitan internet. Dos rondas de revisión adversarial
encontraron defectos reales y ya están corregidos, cada uno con su prueba de
regresión. Algunos ejemplos:

- un paseo aleatorio podía pasar como OU válido. Ahora la validez exige que la
  prueba de Dickey–Fuller aumentada rechace la raíz unitaria, y sobre el spread de
  formación esa prueba es exactamente el paso interno de Engle–Granger;
- un índice sin proxy se simulaba como si fuera operable, y la otra pierna podía
  usarse como su «proxy» (se operaba un activo contra sí mismo);
- la exportación afirmaba «prueba contaminada: no» sin verificarlo, y una nueva
  ejecución de la demo borraba el historial de la prueba.

Las decisiones que tomaron esas correcciones están en
[`docs/DECISIONS.md`](docs/DECISIONS.md) (entre otras, O7, E8, O5a, S10a, S10b,
A2, A2a, A7 y D10–D17).

**Pendiente (fases 2 y 3):** la API, la interfaz web, la configuración de Docker,
el informe HTML autocontenido y la guía del alumno (`docs/GUIA_ALUMNO.md`). Por
ahora, `docker compose up` y `make dev` no están disponibles: el laboratorio se
usa desde Python y desde la demo.

## Requisitos

- **Python 3.11 o superior** (verificado con Python 3.13, pandas 3 y
  statsmodels 0.15).
- `pip` (viene con Python).
- **Git** (opcional, pero recomendado): el laboratorio registra el hash del
  commit en cada `run_config.json`. Sin git se registra «desconocido».
- **make** (opcional): suele venir instalado en Linux; en macOS llega con las
  herramientas de línea de comandos de Xcode (`xcode-select --install`). En
  Windows requiere Git Bash o WSL; si no lo tienes, usa los comandos equivalentes
  de PowerShell que aparecen en cada sección.
- **Internet** solo para descargar datos de Yahoo Finance. Las pruebas y la demo
  por defecto funcionan sin conexión.

## Instalación sin Docker

Desde la carpeta raíz del repositorio, crea un entorno virtual e instala el
paquete en modo editable con las dependencias de desarrollo.

**Linux / macOS:**

```bash
git clone <URL-del-repositorio>
cd Orntein-Uhlenbeck
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e "backend[dev]"
```

**Windows (PowerShell):**

```powershell
git clone <URL-del-repositorio>
cd Orntein-Uhlenbeck
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e "backend[dev]"
```

Si PowerShell no permite ejecutar `Activate.ps1`, habilítalo solo para esa
ventana con `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` y vuelve
a activarlo.

Con `make` disponible y el entorno virtual activado, `make install` hace la misma
instalación (`pip install -e "backend[dev]"`).

### Comandos de make

| Comando | Qué hace | Equivalente sin make |
|---|---|---|
| `make install` | Instala el paquete y las dependencias de desarrollo | `pip install -e "backend[dev]"` |
| `make test` | Ejecuta todas las pruebas | `cd backend` y luego `python -m pytest` |
| `make test-fast` | Ejecuta las pruebas sin las marcadas como lentas | `cd backend` y luego `python -m pytest -m "not slow"` |
| `make demo` | Ejecuta la demo con el par sintético (sin internet) | `python examples/demo_ko_pep.py` |
| `make fetch-example` | Descarga KO y PEP de Yahoo Finance a `examples/data/real/` | `python examples/fetch_ko_pep.py` |
| `make lint` | Revisa el estilo del código con ruff | `python -m ruff check backend examples` |

## Pruebas

```bash
make test
```

Sin `make` (Linux, macOS o PowerShell):

```bash
cd backend
python -m pytest
```

- Las pruebas son deterministas (semillas fijas) y **no necesitan internet**: las
  del proveedor de Yahoo usan respuestas simuladas.
- La suite completa (775 pruebas) tarda unos cinco minutos y medio (5 min 24 s en
  el equipo donde se verificó, con un solo núcleo ocupado). Algo más de dos minutos
  corresponden a las pruebas de la demo, que la ejecutan de punta a punta varias
  veces (también la reproducción byte a byte), y buena parte del resto a las
  pruebas de Monte Carlo de la cointegración y del OU: cada ajuste OU incluye una
  prueba de Dickey–Fuller aumentada con selección de rezagos.
- Dos pruebas están marcadas como lentas (`slow`): el tamaño de la prueba de raíz
  unitaria en las ventanas de estabilidad con 400 pares de paseos aleatorios
  (≈ 45 s) y la robustez del escenario SINT-CBC con 30 semillas (≈ 15 s).
  `make test-fast` (o `python -m pytest -m "not slow"` dentro de `backend`) las
  excluye y tarda alrededor de un minuto menos.

## Demo

La demo `examples/demo_ko_pep.py` reproduce un análisis completo de un par: valida
los datos, ejecuta las pruebas estadísticas, construye el spread, ajusta el OU,
asigna el estado del par con sus causas, simula el backtest y muestra las
métricas.

### Par sintético (por defecto, sin internet)

```bash
make demo
# o, sin make:
python examples/demo_ko_pep.py
```

Sin argumentos, la demo usa el par **SINTÉTICO** de `examples/data/` (generado
con semilla fija y etiquetado como tal). No necesita conexión a internet. La
exportación de resultados se guarda en `examples/output/`, una carpeta que git
ignora (cámbiala con `--output-dir`; la caché de instantáneas, con `--cache-dir`).

Al final, la demo imprime la orden **«Para reproducir»**: vuelve a ejecutar el
`run_config.json` exportado usando solo la caché local y produce los mismos
archivos byte a byte. La orden usa el intérprete de Python con el que ejecutaste
la demo y rutas absolutas (el script, el `run_config.json` y, si no son los
predeterminados, `--cache-dir` y `--output-dir`), más `--excel-es` si lo usaste,
así que funciona desde cualquier carpeta. En Windows se cita con comillas dobles
(en PowerShell, antepón `& `). Tal cual, reescribe la misma carpeta; para comparar,
añade al final `--output-dir <otra carpeta>`. Por ejemplo:

```bash
/ruta/a/.venv/bin/python /ruta/al/repo/examples/demo_ko_pep.py \
    --reproducir /ruta/al/repo/examples/output/<par>/run_config.json
```

### La muestra de prueba

La muestra de prueba (el último 20 %) **no** se ejecuta salvo que lo pidas con
`--incluir-prueba`, y debe ejecutarse una sola vez con la configuración congelada.
Si el `run_config.json` ya incluía la prueba, `--reproducir` la vuelve a ejecutar
sin necesidad de la opción. Si cambias los umbrales a mano y vuelves a ejecutar la
prueba, el informe y la exportación la marcan como **contaminada** («Prueba
contaminada: SÍ»).

En la fase 1, el historial que permite detectarlo viaja dentro del propio
`run_config.json`. La demo lo protege así:

- Antes de exportar, conserva el historial del `run_config.json` que ya haya en la
  carpeta de destino si es del mismo par (también con `--reproducir`). Volver a
  ejecutar la demo normalmente, sin la prueba, ya no lo borra.
- Si ese archivo no se puede leer, o es de otro par y tiene historial, la demo se
  detiene sin sobrescribirlo (`DEMO_HISTORIAL_ILEGIBLE`,
  `DEMO_CARPETA_DE_OTRO_PAR`): usa otra `--output-dir` o, si de verdad quieres
  descartar ese historial, `--reiniciar-historial`.

**Limitación de la fase 1:** la marca no puede detectar una ejecución anterior de
la prueba si borras o editas el historial a mano, si exportas en otra carpeta
(`--output-dir`) o si partes de un `run_config.json` copiado antes de ejecutar la
prueba y exportas fuera de la carpeta original. El registro persistente de
estudios, que lo resuelve, llega en la fase 2.

Si el par no superó los filtros (estado distinto de ✅ o alguna razón de
descarte), el backtest se muestra con el aviso **«resultado solo ilustrativo»** y
sus motivos.

### Tus propios CSV

```bash
python examples/demo_ko_pep.py --csv-a ruta/serie_A.csv --csv-b ruta/serie_B.csv
```

Formato del CSV:

- Columnas `date` y `close` (opcionales: `open`, `high`, `low`, `volume`). También
  se aceptan los encabezados en español `fecha`, `cierre`, `apertura`, `maximo`,
  `minimo` y `volumen`.
- Fechas en formato ISO `AAAA-MM-DD`.
- Separador `,`, `;` o tabulador. Con `;` se acepta coma decimal, como en los CSV
  que exporta Excel en español.
- Debes declarar la **moneda**, la **clase de activo** y el **tipo de precio** de
  cada serie: sin ellos no se puede verificar que ambas estén en la misma moneda ni
  elegir la anualización (252 o 365). Consulta las opciones con
  `python examples/demo_ko_pep.py --help`.
- Si junto al CSV existe `<nombre>.meta.json` (lo crea `make fetch-example`), sus
  metadatos se usan tal cual. Si los metadatos no son válidos, la demo explica en
  español qué campo falla y cómo corregirlo.
- Las dos series deben estar en la **misma moneda** y compartir al menos **834
  fechas** (el 60 % de formación debe tener 500 observaciones o más).
- Una columna opcional `tipo_datos` (HISTORICO o SINTETICO) o un nombre de archivo
  que empieza por `SINTETICO_` marcan los datos como sintéticos.

### Índices y futuros: instrumento ejecutable

Un índice (p. ej. `^GSPC`) o un futuro continuo no se puede comprar ni vender
directamente. Si una serie es un índice, la demo muestra el análisis estadístico
completo, pero **no simula la estrategia** (`PROXY_REQUERIDO`) hasta que indiques
el CSV del instrumento que se opera en su lugar (por ejemplo, un ETF):

```bash
python examples/demo_ko_pep.py --csv-a SPX.csv --csv-b NDX.csv \
    --currency USD --asset-class indice --price-type cierre \
    --exec-a SPY.csv --exec-b QQQ.csv --asset-class-exec-a etf --asset-class-exec-b etf
```

El instrumento ejecutable hereda la moneda y el tipo de precio generales, pero su
clase de activo se declara aparte (`--asset-class-exec-a`/`-b`) o con su
`.meta.json`. La señal sale del índice y el PnL del instrumento ejecutable: la
demo avisa del riesgo de base.

### Opciones de los scripts de `examples/`

Los tres scripts muestran su ayuda y sus errores en español (`-h` o `--help`; un
error de argumentos termina con código 2 y sin trazas de Python).

**`demo_ko_pep.py`** (`python examples/demo_ko_pep.py --help` da la lista completa):

| Opción | Para qué sirve |
|---|---|
| `--csv-a`, `--csv-b` | CSV de las series A y B (sin ellas, el par sintético de `examples/data/`) |
| `--currency`, `--asset-class`, `--price-type` | Metadatos de todas las series, si no hay `.meta.json`; `--currency-a`, `--asset-class-b`, etc. para una sola serie |
| `--symbol-a`, `--symbol-b` | Símbolo de cada serie (por defecto, el nombre del archivo) |
| `--exec-a`, `--exec-b` | CSV del instrumento ejecutable (proxy) de un índice o futuro; su clase con `--asset-class-exec-a`/`-b` (o su `.meta.json`) |
| `--start`, `--end` | Recorta las fechas (AAAA-MM-DD) |
| `--incluir-prueba` | Ejecuta también la muestra de prueba (una sola vez, con la configuración congelada) |
| `--reproducir RUN_CONFIG` | Reproduce un `run_config.json` exportado usando solo la caché (no admite `--csv-*`, `--exec-*`, `--start` ni `--end`) |
| `--reiniciar-historial` | Descarta a conciencia el historial de la prueba que haya en la carpeta de destino |
| `--output-dir CARPETA` | Carpeta de exportación (por defecto, `examples/output`) |
| `--cache-dir CARPETA` | Caché de instantáneas (por defecto, `data/cache`) |
| `--excel-es` | CSV para Excel en español (`;`, coma decimal y BOM) |

**`fetch_ko_pep.py`** (`make fetch-example`, requiere internet):

| Opción | Para qué sirve |
|---|---|
| `--symbols SÍMBOLO …` | Símbolos de Yahoo Finance que se descargan (por defecto, `KO PEP`) |
| `--start AAAA-MM-DD`, `--end AAAA-MM-DD` | Rango de fechas (por defecto, desde 2010-01-01 hasta hoy); `--end` no puede ser anterior a `--start` |
| `--output-dir CARPETA` | Dónde se guardan los CSV y sus `.meta.json` (por defecto, `examples/data/real/`) |

**`generar_csv_sinteticos.py`** regenera el par sintético de `examples/data/`
(byte a byte). Admite un argumento opcional, la carpeta de destino:
`python examples/generar_csv_sinteticos.py [carpeta]`. Una carpeta que empieza por
«-» se rechaza para no confundirla con una opción (escríbela como `./-nombre`).

### Datos reales de KO y PEP

```bash
make fetch-example
python examples/demo_ko_pep.py --csv-a examples/data/real/KO.csv --csv-b examples/data/real/PEP.csv
```

`make fetch-example` descarga Coca-Cola (KO) y PepsiCo (PEP) de Yahoo Finance a
`examples/data/real/` **en tu máquina** (requiere internet). Esos datos no se
incluyen en el repositorio porque los términos de uso de Yahoo desaconsejan
redistribuirlos. KO/PEP es un ejemplo clásico de manual: el laboratorio **no
asume** que esté cointegrado, y el resultado puede ser cualquiera de los cinco
estados.

## Fuentes de datos

### Yahoo Finance (proveedor por defecto)

- **Fuente gratuita y no oficial** (vía la librería `yfinance`), adecuada para uso
  educativo. Puede fallar, limitar las peticiones o cambiar sin aviso. El
  laboratorio reintenta con espera exponencial (1, 2, 4 y 8 segundos) y muestra
  errores claros en español.
- **Precios ajustados por splits y dividendos.** El dividendo ya está dentro del
  rendimiento del precio ajustado, así que el PnL **no** suma dividendos aparte
  (eso los contaría dos veces).
- **Caché inmutable.** Yahoo reajusta todo el histórico con cada dividendo o
  split. Por eso cada descarga se guarda como una instantánea Parquet que no se
  modifica (en `data/cache/`). Reproducir un análisis usa exactamente esas
  instantáneas y nunca vuelve a descargar en silencio.
- **Índices y futuros no son ejecutables.** Si eliges un índice (p. ej. `^GSPC`) o
  un futuro (`GC=F`), el laboratorio sugiere un instrumento ejecutable (`SPY`,
  `GLD`) y te pide confirmarlo; nunca lo sustituye en silencio. Sin instrumento
  ejecutable, el análisis se muestra pero la estrategia no se simula. Ten en
  cuenta el riesgo de base: `^GSPC` no incluye dividendos y `SPY` ajustado sí
  (≈ 1,5 % anual de diferencia).

### CSV propios

Archivos con el formato descrito en [Tus propios CSV](#tus-propios-csv), con la
moneda, la clase de activo y el tipo de precio declarados por ti.

### Datos sintéticos

Pares generados con semilla fija para practicar: cointegrado, cointegrado con baja
correlación, correlacionado pero no cointegrado, y paseos aleatorios
independientes. Sirven para ver, con la respuesta conocida, que una correlación
alta no implica cointegración (y al revés). Siempre llevan la etiqueta
**SINTÉTICO**.

### Añadir otro proveedor

Implementa la interfaz `DataProvider` (`backend/src/mrlab/data/providers/base.py`)
y regístralo en `backend/providers.toml`. Las claves de API se leen de variables de
entorno en el backend y nunca llegan al navegador. La configuración opcional usa
variables con prefijo `MRLAB_` (ver `backend/src/mrlab/config.py`).

## Limitaciones de la v1

- **Solo datos diarios.** Las ejecuciones son al precio de cierre (señal con el
  cierre de `t`, ejecución al cierre de `t+1`); no hay simulación intradía.
- **Misma moneda.** Si los dos activos cotizan en monedas distintas (o la moneda es
  desconocida), el análisis se bloquea. La conversión de moneda queda como
  extensión.
- **Precios positivos** para el spread logarítmico (el que se usa por defecto) y
  siempre para los instrumentos que se operan: con un precio ejecutable ≤ 0 (como
  el petróleo WTI en abril de 2020) el análisis se muestra, pero la estrategia no
  se simula.
- **Mínimo de 500 observaciones** comunes en la muestra de formación.
- **α y β fijos:** se estiman en formación y no se recalibran (la recalibración
  móvil llega en la fase 4).
- **Solo Engle–Granger** para dos activos; Johansen queda como extensión. El error
  estándar y el R² de MCO del hedge ratio se muestran solo como referencia: con
  series cointegradas no sirven para inferencia convencional.
- **Bandas e intervalos sin incertidumbre de parámetros:** tratan λ, μ y σ como
  exactos, así que son más estrechos de lo que deberían. El bootstrap por bloques
  queda como extensión.
- **OU de parámetros constantes:** no modela cambios de régimen. El ajuste OU solo
  es válido si `0 < b < 1` **y** la prueba de Dickey–Fuller aumentada (rezagos por
  AIC) rechaza la raíz unitaria: un `b` algo menor que 1 aparece incluso en un
  paseo aleatorio (ver `docs/MATEMATICAS.md`, sección 8.5). Sobre el spread de un
  par, esa prueba coincide con Engle–Granger sin ajustar. En las ventanas de
  estabilidad, `λ` y la vida media se estiman aunque la prueba no rechace (con 252
  observaciones tiene poca potencia) y el resultado de la prueba se informa
  aparte.
- **Pares cripto + bolsa:** se usan solo los días de negociación de la bolsa (nunca
  se rellenan fines de semana). Lo mismo si un par de cripto se opera con un
  instrumento de bolsa: el calendario es mixto y se anualiza con 252. En Yahoo, el cierre diario de cripto es a
  medianoche UTC y el de la bolsa de EE. UU. a las 16:00 de Nueva York: los dos
  «cierres» no son simultáneos.
- **Cortos en cripto:** vender en corto cripto spot requiere derivados (por
  ejemplo, perpetuos con funding), fuera del alcance de la v1. Sin ventas en corto,
  las señales que necesitan una pierna corta se registran como «señal no
  ejecutable».
- **Costes simplificados:** comisión y slippage fijos en puntos básicos, préstamo
  de cortos y financiación a tasas fijas (todo configurable). No se modelan impacto
  de mercado, ejecuciones parciales, disponibilidad de títulos para préstamo ni
  impuestos. El efectivo no se remunera.
- **Sin calendario de festivos:** los huecos se reportan, pero no se infieren días
  de negociación faltantes.
- **Poca potencia por ventana:** con 252 observaciones, Engle–Granger detecta mal
  incluso relaciones reales, así que muchos pares reales saldrán «inestables».
- **Un par a la vez:** el ranking de un universo con corrección FDR y el
  walk-forward llegan en la fase 4; la cartera multi-par, en la fase 5.

## Estructura del repositorio

```text
.
├── README.md
├── Makefile                       # install, test, demo, fetch-example, lint
├── backend/
│   ├── pyproject.toml             # dependencias, ruff, pytest
│   ├── providers.toml             # registro de proveedores de datos
│   ├── src/mrlab/
│   │   ├── domain/                # modelos compartidos y errores (mensajes en español)
│   │   ├── data/                  # proveedores (Yahoo, CSV, sintético), caché, validación, calendarios, proxies
│   │   ├── stats/                 # correlaciones, ADF/KPSS, Engle–Granger, Benjamini–Hochberg, ventanas
│   │   ├── model/                 # spread (MCO), OU (estimación, pronóstico, simulación), diagnóstico
│   │   ├── strategy/              # parámetros, máquina de estados, dimensionamiento
│   │   ├── backtest/              # división de la muestra, motor, costes, métricas
│   │   ├── analysis/              # análisis de un par (estado y causas), ejecución desde run_config
│   │   ├── reporting/             # run_config.json reproducible, exportación CSV/JSON
│   │   └── api/                   # API FastAPI (fase 2)
│   └── tests/                     # pruebas unitarias y de integración (pytest)
├── frontend/                      # interfaz web React + TypeScript (fase 3)
├── data/                          # cache/, studies/, uploads/: archivos locales, ignorados por git
├── docs/                          # especificación, arquitectura, decisiones, matemáticas
└── examples/
    ├── demo_ko_pep.py             # demo reproducible
    ├── _cli.py                    # argparse en español, común a los tres scripts
    ├── fetch_ko_pep.py            # descarga KO y PEP de Yahoo (make fetch-example)
    ├── generar_csv_sinteticos.py  # regenera examples/data/ (byte a byte)
    └── data/                      # CSV SINTÉTICOS (real/ se crea con make fetch-example)
```

## Documentación

- [`docs/MATEMATICAS.md`](docs/MATEMATICAS.md): todas las fórmulas, con intuición,
  derivación breve y ejemplos numéricos.
- [`docs/DECISIONS.md`](docs/DECISIONS.md): decisiones de diseño, con alternativas
  y justificación.
- [`docs/ARQUITECTURA.md`](docs/ARQUITECTURA.md): módulos, flujo de análisis y
  contratos.
- [`docs/ESPECIFICACION.md`](docs/ESPECIFICACION.md): requisitos originales.
- `docs/GUIA_ALUMNO.md`: recorrido guiado de 15 minutos (pendiente).

---

> **Herramienta educativa. No constituye asesoría de inversión. Resultados
> históricos y simulados no garantizan resultados futuros.**
