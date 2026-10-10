# Registro de decisiones

Cada decisión no especificada en el documento de requisitos se registra aquí con
la decisión, las alternativas consideradas, la justificación y la fecha.
Las decisiones marcadas con **(confirmada)** fueron revisadas explícitamente con
el responsable del proyecto.

---

## Datos

### D1 — Índices y futuros: proxy ejecutable elegido explícitamente (confirmada)
- **Decisión:** los índices (`^…`) y futuros (`…=F`) se marcan como no ejecutables.
  La app sugiere un proxy y ofrece dos modos: (a) **analizar y ejecutar con el
  proxy** (preseleccionado) o (b) analizar el índice y ejecutar con el proxy,
  avisando del riesgo de base (p. ej. `^GSPC` no incluye dividendos y `SPY`
  ajustado sí: ~1,5 % anual de deriva). Nunca se sustituye en silencio.
- **Alternativas:** usar siempre el índice; sustituir automáticamente por el proxy.
- **Justificación:** analizar y operar la misma serie evita que la señal y el PnL
  provengan de instrumentos distintos; el modo (b) se conserva porque la
  especificación lo describe y es pedagógico mostrar el riesgo de base.
- **Fecha:** 2026-10-10

### D2 — Metadatos de Yahoo desde `history_metadata`
- **Decisión:** moneda, tipo de instrumento, nombre, bolsa y zona horaria se leen
  de `history_metadata` de `yfinance`, no de `.info`.
- **Alternativas:** `Ticker.info`; tabla manual de metadatos.
- **Justificación:** `.info` es lento y sufre más bloqueos; `history_metadata`
  llega con la propia descarga de precios.
- **Fecha:** 2026-10-10

### D3 — Reintentos con espera exponencial ante Yahoo
- **Decisión:** 4 intentos con esperas de 1, 2, 4 y 8 s más un pequeño componente
  determinista; mensajes de error en español por tipo (sin datos, límite de
  peticiones, red, respuesta inesperada).
- **Alternativas:** sin reintentos; reintentos infinitos.
- **Justificación:** Yahoo es una API no oficial que limita peticiones; acotar los
  intentos evita bloquear la interfaz indefinidamente.
- **Fecha:** 2026-10-10

### D4 — CSV: formato tolerante pero metadatos obligatorios
- **Decisión:** se detecta el separador (`,` `;` tabulador); con `;` se acepta
  coma decimal; fechas ISO; alias de columnas en español. El usuario **debe**
  declarar moneda, clase de activo y tipo de precio al cargar el archivo.
- **Alternativas:** exigir un único formato; inferir la moneda.
- **Justificación:** los alumnos exportan CSV desde Excel en configuración
  regional española; sin moneda ni clase de activo no se puede validar la moneda
  ni elegir entre 252 y 365 periodos.
- **Fecha:** 2026-10-10

### D5 — Datos sintéticos: etiquetado permanente
- **Decisión:** calendario lunes–viernes sin festivos, símbolos `SINT-*`, nombre
  que empieza por «SINTÉTICO —», `origin = SINTETICO`, prefijo `SINTETICO_` en
  los archivos y columna `tipo_datos` en cada fila exportada. Se tratan como
  acciones a efectos de calendario y costes.
- **Alternativas:** marca solo en la interfaz.
- **Justificación:** prohibición explícita de presentar datos sintéticos como
  históricos; la marca viaja con los datos y no depende de la interfaz.
- **Fecha:** 2026-10-10

### D6 — Calendario: siempre intersección, sin relleno
- **Decisión:** la alineación es siempre la intersección de fechas. El tipo de
  calendario (bolsa, cripto 7 días, mixto) determina Δt, la anualización y la
  detección de huecos. No se usa un calendario de festivos.
- **Alternativas:** `pandas_market_calendars` para detectar festivos faltantes.
- **Justificación:** cumple la política de la sección 2.4 sin dependencias
  adicionales; los faltantes se reportan como NaN del proveedor y fechas presentes
  en una serie y ausentes en la otra.
- **Fecha:** 2026-10-10

### D7 — Instantáneas inmutables direccionadas por contenido
- **Decisión:** cada descarga es una instantánea Parquet inmutable con id
  `proveedor_símbolo_intervalo_hash12`. Nunca se mezclan descargas. Reproducir un
  `run_config.json` usa exactamente esas instantáneas; si falta alguna se informa
  el error y no se vuelve a descargar en silencio.
- **Alternativas:** caché incremental que añade días nuevos a un archivo.
- **Justificación:** Yahoo reajusta todo el histórico con cada dividendo o split;
  añadir días a una descarga antigua mezclaría factores de ajuste distintos.
- **Fecha:** 2026-10-10

### D8 — Bloqueos por datos no aptos usan el estado ⛔
- **Decisión:** monedas distintas, moneda desconocida o precios ≤ 0 con spread
  logarítmico producen el estado «⛔ Datos insuficientes», con la causa concreta
  explicada.
- **Alternativas:** crear un sexto estado «Datos no aptos».
- **Justificación:** la especificación exige exactamente cinco estados; ⛔ es el
  único que bloquea, y la causa detalla el motivo.
- **Fecha:** 2026-10-10

### D9 — Datos de ejemplo: CSV sintéticos en el repositorio (confirmada)
- **Decisión:** `examples/data/` solo contiene CSV **sintéticos**. `demo_ko_pep.py`
  acepta `--csv-a/--csv-b`; `make fetch-example` descarga KO/PEP de Yahoo a CSV en
  la máquina del usuario. Sin argumentos, la demo usa el par sintético.
- **Alternativas:** versionar CSV reales de Yahoo.
- **Justificación:** los términos de uso de Yahoo desaconsejan redistribuir sus
  datos, y la demo debe funcionar sin internet.
- **Fecha:** 2026-10-10

## Estadística

### E1 — Límites del p-value de KPSS
- **Decisión:** KPSS de statsmodels solo interpola entre 0,01 y 0,10; fuera de ese
  rango se muestra «< 0,01» o «> 0,10».
- **Alternativas:** mostrar el valor truncado como si fuera exacto.
- **Justificación:** no presentar precisión inexistente.
- **Fecha:** 2026-10-10

### E2 — Interpretación combinada ADF + KPSS
- **Decisión:** «compatible con I(1)» si en niveles ADF no rechaza y KPSS rechaza,
  y en diferencias ADF rechaza y KPSS no rechaza; «posiblemente estacionaria» si
  en niveles ADF rechaza y KPSS no rechaza; en otro caso «ambiguo». Una serie
  ambigua o posiblemente estacionaria marca el par como «⚠️ Supuestos dudosos».
- **Alternativas:** usar solo ADF.
- **Justificación:** ADF y KPSS tienen hipótesis nulas opuestas; exigir que
  coincidan reduce conclusiones frágiles.
- **Fecha:** 2026-10-10

### E3 — Engle–Granger: dirección y rezagos
- **Decisión:** `statsmodels.tsa.stattools.coint(trend="c", maxlag=12,
  autolag="aic")` en ambas direcciones; por defecto la de menor p-value, indicada
  en la interfaz; el usuario puede fijarla. La dirección elegida define cuál es
  «A» (variable dependiente) en el spread.
- **Alternativas:** una sola dirección; Johansen.
- **Justificación:** EG es asimétrico; Johansen queda como extensión.
- **Fecha:** 2026-10-10

### E4 — Familia para Benjamini–Hochberg
- **Decisión:** BH (q = 0,05) sobre todos los pares × 2 direcciones evaluados; en
  el análisis de un solo par, m = 2. El estado ✅ usa el p-value **ajustado** de la
  dirección elegida. Benjamini–Yekutieli disponible como opción.
- **Alternativas:** Bonferroni; corregir solo por número de pares.
- **Justificación:** BH controla la FDR con más potencia que Bonferroni. Supone
  dependencia positiva entre pruebas (las dos direcciones de un par lo están);
  BY es válido bajo dependencia arbitraria.
- **Fecha:** 2026-10-10

### E5 — Estabilidad: p-values por ventana sin ajustar
- **Decisión:** ventanas de 252 con paso 63 dentro de formación, dirección fija;
  los p-values por ventana no se ajustan.
- **Alternativas:** aplicar BH también por ventana.
- **Justificación:** son un diagnóstico de estabilidad, no un descubrimiento. Con
  252 observaciones EG tiene poca potencia: muchos pares reales saldrán
  «inestables», y se explica en la interfaz.
- **Fecha:** 2026-10-10

### E6 — Precedencia del estado principal
- **Decisión:** ⛔ Datos insuficientes > ⚠️ Supuestos dudosos > ⚠️ Evidencia
  insuficiente > ⚠️ Resultado inestable > ✅. Siempre se listan todas las causas.
- **Alternativas:** mostrar varios estados simultáneos.
- **Justificación:** un estado principal claro para el alumno, sin ocultar causas.
- **Fecha:** 2026-10-10

### E7 — Aviso sobre la inferencia MCO del hedge ratio
- **Decisión:** se muestran el error estándar de β y el R² de MCO con un aviso: con
  series cointegradas la inferencia MCO convencional no es válida
  (superconsistencia) y un R² alto no prueba nada.
- **Alternativas:** ocultarlos; corregirlos con DOLS/FM-OLS.
- **Justificación:** la especificación pide mostrarlos; el aviso evita
  interpretaciones erróneas. DOLS queda como extensión.
- **Fecha:** 2026-10-10

### E8 — Estabilidad del OU sobre el spread operado
- **Decisión:** la estabilidad de λ, μ y σ por ventanas se mide sobre el spread de
  formación con α y β **fijos** (el que se opera), no re-estimando α y β por ventana.
- **Alternativas:** re-estimar el hedge ratio en cada ventana.
- **Justificación:** interesa la estabilidad de la dinámica del spread que
  realmente se opera.
- **Fecha:** 2026-10-10

### E9 — Correlaciones por par sobre fechas comunes
- **Decisión:** los log-rendimientos de cada par se calculan sobre la intersección
  de fechas de sus precios; el número de observaciones se muestra por celda.
- **Alternativas:** calcular rendimientos sobre el calendario propio de cada serie
  y luego alinear.
- **Justificación:** en un par cripto + bolsa, el rendimiento cripto de viernes a
  lunes debe incluir el fin de semana para ser comparable con el de la bolsa.
- **Fecha:** 2026-10-10

## Modelo OU

### O1 — Estimador y error estándar de μ
- **Decisión:** AR(1) por MCO sobre el spread de formación, `s² = SRC/(n−2)`.
  Error estándar asintótico de μ̂: `σ_eq·√((1+b)/((1−b)·n))`.
- **Alternativas:** máxima verosimilitud exacta.
- **Justificación:** con Δt constante, MCO sobre el AR(1) coincide con la MV
  condicional y es transparente para el alumno.
- **Fecha:** 2026-10-10

### O2 — Criterio de la prueba de recuperación de μ (confirmada)
- **Decisión:** con μ = 0 el «error relativo < 10 %» no está definido. Se exige
  `|μ̂ − μ| ≤ 3·EE(μ̂)`. λ (< 15 %) y σ (< 10 %) mantienen el error relativo.
  Se añade una prueba de cobertura con varias semillas.
- **Alternativas:** `|μ̂| < 0,10·σ_eq`.
- **Justificación:** con 5.000 observaciones y vida media ≈ 14 días,
  EE(μ̂) ≈ 0,089·σ_eq. Con semilla 42, |μ̂| = 0,13·σ_eq (1,4 EE). Exigir 0,10·σ_eq
  fallaría en ~27 % de las semillas sin que el estimador tenga error alguno.
- **Fecha:** 2026-10-10

### O3 — Intervalos y diagnóstico
- **Decisión:** intervalos condicionales al 95 % (normal), sin incertidumbre de
  parámetros (aviso visible). Ljung–Box a 10 y 20 rezagos; el puntaje usa 20.
- **Alternativas:** intervalos al 68 %; bootstrap por bloques desde la v1.
- **Justificación:** el 95 % es la convención habitual; el bootstrap queda como
  extensión.
- **Fecha:** 2026-10-10

### O4 — Unidades
- **Decisión:** Δt = 1 observación. «día de negociación» para bolsa y mixtos;
  «día de negociación (calendario de 7 días)» para pares solo cripto. El spread
  logarítmico se expresa en log-puntos (0,01 ≈ 1 %).
- **Alternativas:** años como unidad de tiempo.
- **Justificación:** la vida media en días es directamente interpretable por un
  operador.
- **Fecha:** 2026-10-10

### O5 — Estrategia bloqueada si el OU no es válido; ilustrativa si el par no es ✅ (confirmada)
- **Decisión:** sin un OU válido (b fuera de (0, 1)) no hay σ_eq ni z: la
  estrategia queda bloqueada. Si el OU es válido pero el par no es ✅, el backtest
  se permite con un banner rojo permanente «El par no superó los filtros:
  resultado solo ilustrativo».
- **Alternativas:** bloquear siempre que el par no sea ✅.
- **Justificación:** ver fallar la estrategia en un par sin evidencia es
  pedagógicamente valioso, siempre que no pueda confundirse con evidencia.
- **Fecha:** 2026-10-10

### O6 — Simulación: cruce de μ y vida media empírica
- **Decisión:** se exige X₀ ≠ μ; el cruce se detecta en tiempo discreto (sesgo
  hacia arriba, explicado); la vida media empírica es el primer h en que la media
  de las trayectorias cae a la mitad de (X₀ − μ), con interpolación lineal.
- **Alternativas:** puente browniano para detectar cruces entre observaciones.
- **Justificación:** simple y explicable; la diferencia entre mediana del tiempo
  de cruce y vida media es en sí una lección.
- **Fecha:** 2026-10-10

## Estrategia y backtest

### S1 — Salidas direccionales
- **Decisión:** un corto de spread sale cuando z ≤ +salida; un largo cuando
  z ≥ −salida. Los stops por z también son direccionales (corto: z ≥ +stop;
  largo: z ≤ −stop).
- **Alternativas:** leer «|z| ≤ salida» literalmente.
- **Justificación:** la lectura literal dejaría abierta una posición si z salta de
  +1 a −1, es decir, si el spread revirtió más allá de μ a favor.
- **Fecha:** 2026-10-10

### S2 — Regla de rearme (confirmada)
- **Decisión:** tras una salida por stop (z o monetario) o por tiempo máximo, no se
  reentra por ese lado hasta que z vuelva dentro de ±entrada en algún cierre
  posterior. Configurable; activa por defecto.
- **Alternativas:** sin rearme.
- **Justificación:** sin ella, un stop en z = 3,6 reentra al terminar el
  enfriamiento si z sigue en 3,0, repitiendo la misma operación perdedora.
- **Fecha:** 2026-10-10

### S3 — Enfriamiento
- **Decisión:** solo tras stop (z o monetario). Si la salida se ejecuta en la
  observación e, las señales de entrada de los cierres e..e+4 se ignoran (con 5
  observaciones de enfriamiento); la primera entrada posible se decide en e+5 y
  se ejecuta en e+6, de modo que hay 5 observaciones completas sin posición.
- **Alternativas:** enfriamiento también tras tiempo máximo.
- **Justificación:** la especificación lo define «tras stop»; el rearme cubre el
  tiempo máximo.
- **Fecha:** 2026-10-10

### S4 — Sin reversa en la misma observación; entrada pendiente
- **Decisión:** el estado CERRANDO bloquea entradas. Una entrada decidida y aún no
  ejecutada se representa como SIN_POSICION con orden pendiente (no se crea un
  sexto estado).
- **Alternativas:** permitir salir y entrar en el lado contrario en la misma barra.
- **Justificación:** simplicidad y fidelidad a los cinco estados especificados.
- **Fecha:** 2026-10-10

### S5 — Nocional decidido en t, cantidades al precio de t+1
- **Decisión:** el nocional objetivo se calcula con el capital del cierre de la
  señal (t); las cantidades se obtienen con el precio de ejecución (cierre t+1),
  como una orden al cierre por nocional.
- **Alternativas:** cantidades con el precio de t.
- **Justificación:** mantiene la exposición objetivo exacta en la ejecución sin
  usar información posterior a la decisión.
- **Fecha:** 2026-10-10

### S6 — Stop monetario
- **Decisión:** se evalúa al cierre t con el PnL neto acumulado de la operación
  (incluidos costes) frente al 5 % del capital inicial del par, y se ejecuta en
  t+1. La pérdida real puede superar el 5 % (gaps); la interfaz lo avisa.
- **Alternativas:** ejecutar el stop al mismo cierre t.
- **Justificación:** coherencia con la convención de ejecución t+1 (sección 8.2).
- **Fecha:** 2026-10-10

### S7 — Apalancamiento y redondeo
- **Decisión:** el límite de 2× se verifica en la entrada (si el objetivo lo
  supera, se reduce y se registra). Si durante la operación se supera, se reduce
  proporcionalmente en la observación siguiente, con costes, y se registra. El
  redondeo a enteros es siempre hacia cero; si una pierna queda en 0 no se opera.
- **Alternativas:** liquidar la posición completa al superar el límite.
- **Justificación:** intervención mínima y trazable.
- **Fecha:** 2026-10-10

### S8 — Segmentos independientes y fechas de corte congeladas
- **Decisión:** formación (ilustrativa, dentro de muestra), validación y prueba
  se simulan por separado, cada una empezando sin posición con el capital
  inicial y usando los parámetros de formación. Una posición abierta al final
  queda como no realizada y marcada. Las fechas de corte 60/20/20 se congelan en
  el `RunConfig`.
- **Alternativas:** una simulación continua con informes por tramo.
- **Justificación:** la prueba debe ejecutarse una vez y de forma aislada; z solo
  depende de los parámetros fijos, por lo que no hace falta calentamiento.
- **Fecha:** 2026-10-10

### S9 — Señales sin ventas en corto
- **Decisión:** con cortos desactivados y β > 0 ninguna señal es ejecutable; con
  β < 0, «comprar el spread» (dos largos) sí lo es. Las no ejecutables se
  registran y se muestran como «señal no ejecutable».
- **Alternativas:** ejecutar solo la pierna larga.
- **Justificación:** una sola pierna no es una estrategia de pares.
- **Fecha:** 2026-10-10

### S10 — Prueba contaminada
- **Decisión:** el estado del estudio se guarda en `data/studies/`. Ejecutar la
  prueba registra el hash de la configuración; cualquier cambio posterior y nueva
  ejecución marca «prueba contaminada» de forma permanente, también en las
  exportaciones.
- **Alternativas:** solo un aviso en la sesión del navegador.
- **Justificación:** la marca debe sobrevivir a recargas y viajar con los resultados.
- **Fecha:** 2026-10-10

## Costes y métricas

### C1 — Comisión y slippage como coste en pb
- **Decisión:** se cobran sobre el nocional operado, por pierna; el slippage
  depende de la clase de activo de cada pierna. No se altera el precio de
  ejecución.
- **Alternativas:** desplazar el precio de ejecución.
- **Justificación:** contabilidad más transparente: el coste aparece separado del
  PnL bruto.
- **Fecha:** 2026-10-10

### C2 — Devengo por días naturales; efectivo sin remunerar
- **Decisión:** préstamo de cortos y financiación se devengan por días naturales
  / 365 (el lunes devenga el fin de semana). Monto financiado = max(0, exposición
  bruta − capital). El efectivo no se remunera.
- **Alternativas:** devengo por observación / 252.
- **Justificación:** así se cobran en la práctica; coherente con la tasa libre de
  riesgo 0 del Sharpe.
- **Fecha:** 2026-10-10

### C3 — Realizado y no realizado
- **Decisión:** realizado = PnL neto de operaciones cerradas; no realizado = PnL
  bruto marcado a mercado de la operación abierta menos los costes ya incurridos
  por ella. Todos los costes se atribuyen a una operación.
- **Alternativas:** costes de mantenimiento como partida aparte.
- **Justificación:** garantiza realizado + no realizado = PnL total.
- **Fecha:** 2026-10-10

### C4 — Convenciones de métricas
- **Decisión:** Sharpe con rendimientos diarios del capital incluyendo días sin
  posición, rf = 0, 252 o 365 (solo cripto). Factor de beneficio sin pérdidas: «no
  definido». Costes como % del PnL bruto: «no aplica» si el bruto ≤ 0.
- **Alternativas:** Sharpe solo sobre días con posición.
- **Justificación:** refleja el rendimiento del capital asignado al par.
- **Fecha:** 2026-10-10

## Ranking y cartera (fases 4 y 5)

### R1 — Puntaje compuesto
- **Decisión:** puntaje = Σ wᵢ·sᵢ con sᵢ ∈ [0, 1], fórmulas visibles:
  cointegración `clip(1 − p_aj/0,10)`; estabilidad `1 − fracción de ventanas
  fallidas` (0 si β cambia de signo); vida media 1 en [5, 63] días, lineal hasta 0
  en 1 y 126; calidad OU `½·min(1, p_LB/0,05) + ½·(1 − min(1, CV de λ))`; costes
  `clip(1 − coste ida y vuelta / ((z_entrada − z_salida)·σ_eq))`; validación
  `clip(Sharpe_validación/2)` (×½ si hay menos de 10 operaciones). Pesos por
  defecto 0,25 / 0,20 / 0,15 / 0,15 / 0,10 / 0,15. Los pares no ✅ o con OU
  inválido van a una tabla de descartados con sus causas. La prueba va en una
  columna aparte y no entra en el puntaje.
- **Alternativas:** ordenar por p-value; ordenar por PnL.
- **Justificación:** la especificación prohíbe ordenar por PnL retrospectivo.
- **Fecha:** 2026-10-10

### R2 — Walk-forward
- **Decisión:** la posición se cierra al final de cada ventana de prueba; si en una
  ventana el OU es inválido, no se opera en ella y se registra.
- **Alternativas:** arrastrar la posición entre ventanas.
- **Justificación:** ventanas independientes y auditables.
- **Fecha:** 2026-10-10

### P1 — Cartera: el consolidado es la suma de pares
- **Decisión:** los pares se simulan juntos bajo límites comunes; el PnL
  consolidado es exactamente la suma de los PnL por par. La compensación de
  posiciones compartidas se informa como exposición neta y «ahorro potencial»,
  pero no se aplica al PnL en la v1.
- **Alternativas:** compensar posiciones y cobrar costes sobre el neto.
- **Justificación:** el criterio de terminado exige que la curva consolidada
  cuadre con la suma de pares.
- **Fecha:** 2026-10-10

## Ingeniería

### I1 — Paquete y dependencias
- **Decisión:** paquete Python `mrlab` (en `backend/src`). Verificado con
  Python 3.13, pandas 3.0 y statsmodels 0.15; la imagen Docker usa Python 3.12.
- **Alternativas:** fijar pandas 2.2.
- **Justificación:** statsmodels 0.15 funciona con pandas 3.
- **Fecha:** 2026-10-10

### I2 — Trabajos largos en proceso
- **Decisión:** gestor de trabajos dentro del proceso y consulta periódica de
  `/api/jobs/{id}`; sin Redis ni Celery.
- **Alternativas:** Celery + Redis.
- **Justificación:** uso local, máximo 30 activos; menos piezas que instalar.
- **Fecha:** 2026-10-10

### I3 — Informe HTML generado en el backend
- **Decisión:** el backend genera el informe con Plotly y embebe plotly.js
  (~4,5 MB) para que sea autocontenido y funcione sin internet.
- **Alternativas:** generarlo en el navegador; cargar plotly.js desde un CDN.
- **Justificación:** reproducible, comprobable con pruebas y compartible offline.
- **Fecha:** 2026-10-10

### I4 — Exportación determinista
- **Decisión:** floats con `format(x, ".12g")`, claves JSON ordenadas, LF, ZIP con
  fechas fijas, sin marcas de tiempo de generación. Opción «CSV para Excel en
  español» (`;` y coma decimal).
- **Alternativas:** marcas de tiempo de exportación en cada archivo.
- **Justificación:** requisito de reproducibilidad byte a byte.
- **Fecha:** 2026-10-10

### I5 — Identidad de la configuración
- **Decisión:** `config_hash` = sha256 del JSON canónico del `RunConfig` sin
  `config_hash` ni `code_version`. La versión del código (hash de git, `-dirty` si
  hay cambios; variable `GIT_COMMIT` en Docker) se registra aparte.
- **Alternativas:** incluir la versión del código en el hash.
- **Justificación:** la misma configuración debe identificarse igual aunque cambie
  el código; la versión se audita por separado.
- **Fecha:** 2026-10-10

### I6 — Windows, idioma, puertos
- **Decisión:** `make` requiere Git Bash o WSL en Windows, por lo que el README da
  los comandos equivalentes en PowerShell. Español neutro latinoamericano;
  números `10.000,5` y fechas ISO en tablas. Backend en `:8000`, frontend (nginx)
  en `:8080`.
- **Alternativas:** script de tareas en Python.
- **Justificación:** Docker funciona igual en los tres sistemas; los comandos
  directos son transparentes.
- **Fecha:** 2026-10-10

## Integración de la fase 1

### A1 — Causas del estado y razones de descarte
- **Decisión:** `causes` contiene TODAS las causas activas del estado principal,
  ordenadas por la precedencia E6; si no hay ninguna, el estado es ✅ y lleva una
  única causa con el p-value ajustado de Engle–Granger. La validez del OU no es
  un estado: un OU no válido (b fuera de (0, 1)) bloquea la estrategia y aparece
  en «¿Por qué se descartó?» sin cambiar el estado. Lo mismo para la vida media
  fuera de [1, 126], b indistinguible de 1 (`near_unit_root`), el filtro de
  correlación y «costes superan el PnL bruto» en validación. Un error de datos
  durante los cálculos (p. ej. una serie constante en formación) da ⛔ con la
  causa `CALCULO_NO_POSIBLE`.
- **Alternativas:** convertir el OU no válido en un sexto estado; mostrar solo la
  causa que decide el estado.
- **Justificación:** la especificación fija exactamente cinco estados y pide
  listar todas las causas; separar estado (evidencia estadística) y descartes
  (aptitud para operar) es más claro para el alumno.
- **Fecha:** 2026-10-10

### A2 — Proxy ejecutable: alineación por intersección
- **Decisión:** con `exec_a`/`exec_b`, el análisis y el backtest usan solo las
  fechas en que cotizan también los instrumentos ejecutables (nunca se rellena) y
  se avisa cuántas fechas se excluyeron y del riesgo de base. El proxy debe estar
  en la misma moneda que la serie analizada (si no, ⛔). Sin proxy, un índice o
  futuro se opera sobre su propia serie con un aviso que sugiere el proxy.
- **Alternativas:** reindexar el proxy y rellenar huecos; bloquear índices sin proxy.
- **Justificación:** coherente con D1 y D6 (nunca sustituir en silencio, nunca
  rellenar); el aviso mantiene el valor pedagógico del modo (b) de D1.
- **Fecha:** 2026-10-10

### A3 — `run_config.json` autosuficiente
- **Decisión:** `run_pair` devuelve el RunConfig completado: cada referencia
  queda fijada a la instantánea usada (`snapshot_id`, conservando las fechas
  pedidas), `frozen_split`, `series`, `data_origin` según los datos (SINTETICO si
  alguna serie lo es, aunque la configuración dijera otra cosa), `code_version` y
  `config_hash`. Sin `snapshot_id` la referencia debe traer `start` y `end`. Una
  instantánea cuyo proveedor o símbolo no coincide con la referencia se rechaza.
  Un `run_config.json` escrito a mano sin `config_hash` se acepta y se le calcula;
  uno con hash que no coincide se rechaza.
- **Alternativas:** guardar solo la configuración de entrada y resolver los
  datos en cada ejecución.
- **Justificación:** guardar, cargar y volver a ejecutar debe dar los mismos
  bytes sin tocar el proveedor (D7, sección 15).
- **Fecha:** 2026-10-10

### A4 — Formato de la exportación
- **Decisión:** `tipo_datos` es la primera columna de cada CSV. Con datos
  sintéticos TODOS los archivos (también `run_config.json` y `LEEME.txt`) y la
  carpeta sugerida llevan el prefijo `SINTETICO_`. `pruebas_estadisticas.csv` va
  en formato largo (`seccion, elemento, campo, valor`) e incluye estado, causas,
  descartes y métricas por segmento. Booleanos «sí»/«no»; fracciones como
  fracción; `-0` se escribe `0`. La opción «CSV para Excel en español» añade el
  BOM de UTF-8. El ZIP fija fecha, sistema de origen y permisos de cada entrada.
- **Alternativas:** prefijo solo en los CSV; una columna por prueba.
- **Justificación:** la marca de datos sintéticos debe viajar con cualquier
  archivo suelto (D5); sin BOM, Excel en Windows no muestra bien las tildes; un
  formato largo admite pruebas con columnas distintas.
- **Fecha:** 2026-10-10

### A5 — Versión del código «-dirty»
- **Decisión:** `-dirty` solo si hay archivos versionados modificados
  (`git status --porcelain --untracked-files=no`), como `git describe --dirty`.
- **Alternativas:** contar también los archivos sin versionar.
- **Justificación:** las salidas y la caché locales no deben cambiar la versión
  registrada.
- **Fecha:** 2026-10-10

### A6 — z móvil en el backtest
- **Decisión:** con `zscore_mode="movil"` el runner calcula el z móvil sobre el
  spread completo (ventana hacia atrás que incluye t) y cada backtest lleva el
  aviso de que es solo una comparación pedagógica. `PairAnalysis.zscore` es
  siempre el z con μ y σ_eq fijos de formación.
- **Alternativas:** recalcular el z móvil dentro de cada segmento (perdería las
  primeras observaciones de validación y prueba).
- **Justificación:** usa solo información pasada y no desperdicia datos.
- **Fecha:** 2026-10-10

### A7 — Demo reproducible
- **Decisión:** `examples/demo_ko_pep.py` lee los metadatos de `<archivo>.meta.json`
  si existe (lo crean `make fetch-example` y el generador sintético); si no, exige
  `--currency`, `--asset-class` y `--price-type` (D4). Usa la caché `data/cache`
  (o `--cache-dir`), ejecuta la prueba solo con `--incluir-prueba` y admite
  `--reproducir <run_config.json>` (solo caché, nunca descarga).
- **Alternativas:** caché temporal en cada ejecución.
- **Justificación:** con una caché persistente el `run_config.json` exportado se
  puede reproducir más tarde.
- **Fecha:** 2026-10-10

### A8 — Umbral de evidencia y q de la FDR
- **Decisión:** la evidencia se decide con el p-value AJUSTADO de la dirección
  elegida frente a `alpha_level` (rechazo si p ≤ α). Los p-values ajustados de
  BH/BY no dependen de `fdr_q`; por defecto ambos valen 0,05.
- **Alternativas:** decidir con el conjunto rechazado por BH a nivel q.
- **Justificación:** el usuario ve un único umbral, el mismo que en la interfaz.
- **Fecha:** 2026-10-10

### A9 — Decisiones de implementación de los módulos del núcleo
- **Decisión:** se registran aquí las tomadas al implementar cada módulo:
  - Yahoo: `yfinance` configurado para LANZAR sus errores
    (`hide_exceptions = False`); si no, un fallo de red llega como tabla vacía y
    se confundiría con «símbolo sin datos». `max_retries` son reintentos tras el
    primer intento: con 4, hasta 5 llamadas y esperas de 1, 2, 4 y 8 s (D3).
  - Caché: el hash de la instantánea combina el contenido con la identidad de la
    serie (activo, moneda, tipo de precio, intervalo y rango pedido); la fecha de
    descarga no entra en el hash.
  - CSV: se relee en cada carga (el usuario puede subir un archivo corregido con
    el mismo símbolo; cada versión queda como instantánea propia); si no es UTF-8
    válido se interpreta como Windows-1252 (Excel en español).
  - OU: `near_unit_root` (Dickey–Fuller al 5 % con constante) es solo un aviso;
    la validez sigue siendo 0 < b < 1 (O5). Spread constante o determinista →
    `INSUFFICIENT`. Con NaN en el spread solo se usan pares consecutivos finitos.
    Ljung–Box con `model_df=1`. Simulación con horizonte máximo de 2.520 pasos.
  - Backtest: salida forzada (`ExitReason.FORCED`, máxima prioridad, sin
    enfriamiento ni rearme) si el capital se agota con posición abierta o si una
    reducción de apalancamiento con unidades enteras dejaría una pierna en 0. La
    reducción por apalancamiento solo se dispara si el límite se supera en más
    de un 1 %.
- **Alternativas:** documentarlas solo en los docstrings.
- **Justificación:** son decisiones no especificadas que afectan resultados o
  mensajes; deben quedar auditables en un único lugar.
- **Fecha:** 2026-10-10
