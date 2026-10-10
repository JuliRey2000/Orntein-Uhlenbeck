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
  ajustado sí: ~1,5 % anual de deriva). Nunca se sustituye en silencio. Sin un
  proxy elegido, el análisis estadístico se muestra completo, pero la simulación
  queda bloqueada (`PROXY_REQUERIDO`; ver A2).
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
  Un CSV cargado por el usuario también se marca como SINTÉTICO si cualquier
  indicio lo dice (D10).
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
  el error y no se vuelve a descargar en silencio. El contenido se verifica al
  leer (D12).
- **Alternativas:** caché incremental que añade días nuevos a un archivo.
- **Justificación:** Yahoo reajusta todo el histórico con cada dividendo o split;
  añadir días a una descarga antigua mezclaría factores de ajuste distintos.
- **Fecha:** 2026-10-10

### D8 — Bloqueos por datos no aptos usan el estado ⛔
- **Decisión:** monedas distintas, moneda desconocida (también la del proxy
  ejecutable: `MONEDA_DESCONOCIDA`, nunca «None» en el mensaje), precios ≤ 0 con
  spread logarítmico y el mismo activo en las dos piernas (`MISMO_ACTIVO`: mismo
  proveedor y mismo símbolo, sin distinguir mayúsculas ni espacios) producen el
  estado «⛔ Datos insuficientes», con la causa concreta explicada. El mismo
  símbolo de proveedores distintos no es ⛔, pero queda bloqueado por
  colinealidad (O8). Si lo que coincide son los instrumentos que se OPERAN (tras
  los proxies ejecutables), el estado no cambia y la simulación queda bloqueada
  con `EJECUTABLES_IDENTICOS` (A2a).
- **Alternativas:** crear un sexto estado «Datos no aptos»; para el mismo activo,
  solo un aviso.
- **Justificación:** la especificación exige exactamente cinco estados; ⛔ es el
  único que bloquea, y la causa detalla el motivo. Elegir dos veces el mismo
  activo es un error plausible de un alumno y ninguna prueba tiene sentido.
- **Fecha:** 2026-10-10

### D9 — Datos de ejemplo: CSV sintéticos en el repositorio (confirmada)
- **Decisión:** `examples/data/` solo contiene CSV **sintéticos**. `demo_ko_pep.py`
  acepta `--csv-a/--csv-b`; `make fetch-example` descarga KO/PEP de Yahoo a CSV en
  la máquina del usuario. Sin argumentos, la demo usa el par sintético. El
  generador (`examples/generar_csv_sinteticos.py`) escribe la columna
  `tipo_datos = SINTETICO` en cada fila y la semilla (`seed`) en el `.meta.json`;
  una prueba comprueba que los archivos versionados coinciden byte a byte con lo
  que genera.
- **Alternativas:** versionar CSV reales de Yahoo.
- **Justificación:** los términos de uso de Yahoo desaconsejan redistribuir sus
  datos, y la demo debe funcionar sin internet.
- **Fecha:** 2026-10-10

### D10 — CSV: cualquier indicio de datos sintéticos los marca como SINTÉTICOS
- **Decisión:** una serie CSV es SINTÉTICA si lo dicen los metadatos (`origin` o
  `price_type`), si declaran una semilla (`seed`), si alguna fila de la columna
  opcional `tipo_datos` dice SINTETICO (sin distinguir mayúsculas ni tildes; las
  celdas vacías no informan) o si el nombre del archivo empieza por `SINTETICO_`
  (al registrarlo con su nombre original o al leer un archivo guardado con ese
  prefijo). Al forzarlo se añade una nota explicativa (`AssetInfo.notes`) y el
  `.meta.json` se guarda con los metadatos efectivos; la demo lo avisa. Una
  columna HISTORICO nunca rebaja datos declarados sintéticos (se anota). Filas
  HISTORICO y SINTETICO mezcladas (`CSV_TIPO_DATOS_MEZCLADO`) o valores
  desconocidos (`CSV_TIPO_DATOS_DESCONOCIDO`) son un error. El nombre nunca prueba
  que los datos sean históricos.
- **Alternativas:** rechazar con error cuando los metadatos contradicen la columna
  o el nombre; ignorar el nombre del archivo; tratar las celdas vacías como mezcla.
- **Justificación:** la sección 17 y D5 prohíben presentar datos sintéticos como
  históricos y piden que la marca viaje con los datos. Antes, copiar solo el CSV
  de ejemplo y declararlo como cierre ajustado lo exportaba como HISTORICO.
  Forzar en la dirección segura, con una nota, no interrumpe al alumno y nunca
  etiqueta mal; la mezcla sí se rechaza porque una serie no puede ser real y
  generada a la vez.
- **Fecha:** 2026-10-10

### D11 — Metadatos CSV con errores en español
- **Decisión:** `parse_metadata(dict | str | bytes)` y `build_metadata(**campos)`
  convierten cualquier fallo (JSON inválido, con línea y columna; bytes que no son
  UTF-8; campo que falta; valor fuera de la lista; tipo incorrecto) en
  `DataError CSV_METADATOS_INVALIDOS`. El mensaje nombra el campo, el valor
  recibido y los valores válidos, y añade una ayuda con los campos obligatorios.
  La moneda se valida solo como formato ISO de 3 letras. Clase de activo, tipo de
  precio y origen se aceptan sin distinguir mayúsculas ni tildes («Acción»). Un
  `.meta.json` dañado no impide usar los demás archivos, pero se nombra en el
  error `SIMBOLO_SIN_DATOS`. La demo y el script de descarga usan estas funciones.
- **Alternativas:** pasar el mensaje de pydantic tal cual; validar la moneda
  contra la lista ISO 4217 completa.
- **Justificación:** la sección 0 exige errores claros en español que digan qué
  hacer (antes la demo terminaba con un traceback de pydantic en inglés). Para
  comprobar que dos monedas son iguales basta el código, y una lista ISO habría que
  mantenerla.
- **Fecha:** 2026-10-10

### D12 — Instantáneas verificadas al leer (complementa D7 y A9)
- **Decisión:** `SnapshotCache` recalcula el id de cada instantánea al leerla,
  con el mismo cálculo que al guardarla, sobre los datos leídos y la identidad tal
  como quedó guardada en el JSON del Parquet (no sobre el modelo revalidado). Si no
  coincide con el id pedido o con el de sus metadatos → `SNAPSHOT_CORRUPTO`, con
  instrucciones para regenerarla; nunca se usa en silencio. Guardar de nuevo una
  instantánea alterada o ilegible la reescribe: volver a cargar el CSV original o
  regenerar los datos sintéticos la repara. `DataService.load` no vuelve a
  descargar automáticamente: informa el error y hay que pedir la actualización.
  `SeriesMeta.seed` no entra en la identidad, así que los ids existentes no
  cambian, y un campo nuevo con valor por defecto tampoco invalida instantáneas
  antiguas.
- **Alternativas:** verificar solo al guardar; un archivo de sumas de control
  aparte; recalcular sobre el modelo revalidado (rompería la caché al añadir
  campos); volver a descargar en silencio.
- **Justificación:** D7 exige que reproducir use exactamente esas instantáneas.
  Con un id direccionado por contenido, verificarlo al leer es la única garantía
  real (antes una instantánea alterada se usaba en silencio, incluso recargando el
  CSV original).
- **Fecha:** 2026-10-10

### D13 — Parámetros del escenario SINT-CBC (cointegrado con baja correlación)
- **Decisión:** volatilidad diaria de B 0,01, OU con `σ = 0,03` y `λ = 0,5`; la
  deriva de B sigue en 0,0002 y la semilla del preset en 7. Teoría:
  `σ_eq = σ/√(2λ) = 3 %`; `Var(ΔX) = 2·σ_eq²·(1 − e^{−λ}) ≈ 0,000708` (≈ 2,66 %
  diario); `corr(r_A, r_B) = 0,01/√(0,01² + 0,000708) ≈ 0,35`; vida media
  `ln 2/0,5 ≈ 1,39` días. Con la semilla 7: A y B compatibles con I(1),
  Engle–Granger p ≈ 5·10⁻²⁷, correlación en formación 0,33, estado ✅,
  `λ̂ ≈ 0,53` (vida media estimada 1,31 días). Robustez: 22 de 30 semillas ✅
  (antes 10 de 30) y 75 de 90 en las semillas 1–90, todas con correlación < 0,5
  (media 0,35, máximo 0,43). Casi todas las fallas vienen de que el paseo
  aleatorio B sale «ambiguo» con ADF + KPSS (≈ 10 % de los paseos aleatorios con
  la regla E2 al 5 %), lo que ningún parámetro del OU corrige.
- **Alternativas:** los valores anteriores (volatilidad de B 0,004, `σ = 0,015`,
  `λ = 0,1`: la serie A parecía estacionaria y el preset daba «⚠️ Supuestos
  dudosos»; 10 de 30); `λ = 1` (26 de 30, pero vida media < 1 día, fuera del rango
  operable); deriva 0,0006–0,001 (sin mejora en 90 semillas y la semilla 7 falla
  desde 0,0008); `σ = 0,025` (correlación máxima 0,49, sin margen).
- **Justificación:** el ejemplo pedagógico debe mostrar de verdad «cointegrado con
  baja correlación» con ✅ y sin supuestos dudosos. Un OU rápido domina el cambio
  diario (correlación baja) sin que la serie A parezca estacionaria en niveles.
  Cifras verificadas con el código (MATEMATICAS §2.2).
- **Fecha:** 2026-10-10

### D14 — Umbral de salto sospechoso simétrico en logaritmos
- **Decisión:** `SALTO_SOSPECHOSO` si `|ln(P_t/P_{t−1})| > ln 1,5`, es decir, una
  subida de más del 50 % o una caída de más del 33,3 % en un día. El mensaje sigue
  hablando de un posible error de ajuste por split o dividendo y menciona que un
  split 2:1 sin ajustar aparece como una caída del 50 %.
- **Alternativas:** `|R simple| > 50 %` (anterior: asimétrico; no marcaba el split
  2:1 sin ajustar, el error de ajuste más común); marcar `R > 0,5` o `R ≤ −0,45`.
- **Justificación:** un split k:1 sin ajustar y su contrasplit tienen el mismo
  `|ln|` (`ln k`): el umbral en logaritmos los trata igual.
- **Fecha:** 2026-10-10

### D15 — Resumen de validación con errores no bloqueantes
- **Decisión:** el resumen dice «Datos aptos para el análisis…» solo si ninguna
  incidencia tiene severidad error. Si el par no está bloqueado pero hay errores
  (p. ej. precios ≤ 0 con spread en precios), dice «Datos utilizables con
  incidencias: <cada error con símbolo y número de fechas>; además hay N
  advertencia(s)…» y pide revisarlas antes de confiar en el análisis. Las
  incidencias de los instrumentos ejecutables se añaden al informe y el resumen
  las menciona (A2).
- **Alternativas:** mantener «Datos aptos» y contar los errores como advertencias;
  bloquear siempre que haya errores.
- **Justificación:** el resumen no puede contradecir una incidencia marcada como
  error. Bloquear el spread en precios con precios ≤ 0 iría más allá de la sección
  2.3; lo que no se puede simular lo bloquea el análisis (S12).
- **Fecha:** 2026-10-10

### D15a — Resumen de validación con las fechas realmente usadas (complementa D15 y A2)
- **Decisión:** con instrumentos ejecutables (o con un `frozen_split` cuyos
  conteos difieran), el informe de validación se recalcula con las fechas
  realmente usadas: `n_aligned` = observaciones tras la intersección con los
  ejecutables, `n_train_estimate` = formación efectiva, y `status_message` con
  `validation.usable_status` (antes `_usable_status`) sobre TODAS las incidencias.
  Las incidencias de un ejecutable llevan `details["instrumento_ejecutable"] =
  True`; su error se resume como «… en el instrumento ejecutable «X» (n fecha(s)),
  que bloquean la simulación» y se añade «Incluye N incidencia(s) de los
  instrumentos ejecutables». El informe queda bloqueado («Análisis bloqueado: …»,
  `blocked = True`, con su incidencia) si la formación queda por debajo del
  mínimo tras la intersección (`DATOS_INSUFICIENTES`) o si la moneda del proxy es
  distinta o desconocida (`MONEDA_PROXY_DISTINTA`, `MONEDA_DESCONOCIDA`). Si nada
  cambia respecto de `validate_pair`, el informe se conserva idéntico.
- **Alternativas:** conservar el resumen de `validate_pair` y añadir una frase
  sobre los ejecutables (versión anterior).
- **Justificación:** con un proxy que empezaba más tarde, el resumen decía
  «1.500 observaciones comunes, 900 en formación» cuando el análisis usaba 900 y
  540; con un precio del proxy ≤ 0 (error que bloquea la simulación) decía «Datos
  aptos». Ambas cosas contradicen D15 y la sección 2.3.
- **Fecha:** 2026-10-10

### D16 — Logger de yfinance (complementa A9)
- **Decisión:** al importar `yfinance`, el logger «yfinance» sube a ERROR (solo
  se sube: si alguien lo fijó más alto se respeta), salvo que esté activo el modo
  de depuración de yfinance (`yf.config.debug.logging`).
- **Alternativas:** redirigir el logger a un archivo; nivel CRITICAL; no tocarlo.
- **Justificación:** con `hide_exceptions = False` los fallos ya llegan como
  excepciones que el laboratorio traduce; los avisos en inglés (p. ej. «Cookie
  fetch from fc.yahoo.com failed») solo duplicaban información antes del mensaje
  en español. El modo de depuración sigue disponible para diagnosticar.
- **Fecha:** 2026-10-10

### D17 — Calendario efectivo: todos los instrumentos usados (complementa D6 y A2)
- **Decisión:** el tipo de calendario se determina con las clases de TODAS las
  series usadas: las analizadas y las ejecutables. Cripto analizado con un
  ejecutable de bolsa da calendario mixto: 252 periodos, unidad «día de
  negociación», incidencia `DESFASE_CIERRE_CRIPTO` y aviso «Calendario efectivo
  mixto». `PairAnalysis.calendar` y el calendario del informe de validación
  reflejan ese calendario efectivo.
- **Alternativas:** usar solo las series analizadas.
- **Justificación:** tras alinear con un proxy de bolsa, los datos solo tienen días
  hábiles; anualizar con 365 inflaba el Sharpe ≈ 20 % y rotulaba mal la unidad del
  OU.
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
  los p-values por ventana no se ajustan. Una ventana no evaluable cuenta como
  fallida (E10).
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
- **Actualización (2026-10-10), parámetros y prueba de raíz unitaria por
  ventana:** λ, μ, σ, la vida media y CV(λ) se calculan en toda ventana con
  `0 < b̂ < 1` (`fit_ou(..., require_unit_root_rejection=False)`;
  `OUWindow.valid` pasó a llamarse `OUWindow.estimable` y se añade `b`). La
  prueba ADF de cada ventana se informa APARTE y no filtra los parámetros:
  `unit_root_rejected`, `df_stat`, `df_pvalue`, `df_lags`, más el resumen
  `n_unit_root_tested`, `n_unit_root_rejected` y `unit_root_alpha`. Usa los
  valores críticos de una serie observada (N = 1), porque α y β se estimaron con
  toda la formación y no se re-estiman en la ventana. `OUStability.n_invalid`
  cuenta las ventanas sin parámetros. El mensaje da la mediana de la vida media,
  cuántas ventanas rechazan, la baja potencia con 252 observaciones y aclara que
  no rechazar en una ventana no prueba inestabilidad: la validez del OU la decide
  el ajuste de toda la formación (O7).
  - *Alternativas:* aplicar en cada ventana la regla completa de O7 con N = 2
    (versión anterior); aplicarla con N = 1.
  - *Justificación:* con N = 2 en tramos de un residuo con α y β fijos, la prueba
    solo rechazaba el 1,6 % de las ventanas de paseos independientes (5 %
    nominal). Con OU verdaderos (λ = 0,05), el 83 % de las ventanas quedaba «no
    válido», el 48 % de los pares no tenía CV(λ) y, como solo pasaban las
    ventanas con λ alto, la mediana de la vida media salía 6,6 días frente a
    13,9 verdaderos. Exigir el rechazo con N = 1 seguiría seleccionando por λ.
    Medido con la regla actual (MATEMATICAS §8.7): con 400 pares de paseos
    independientes, el ADF de la ventana rechaza en el 7,2 % de las 4.400
    ventanas (algo más del 5 % porque α y β se estimaron con toda la formación y
    por la selección de rezagos en muestras cortas). Con 100 pares OU verdaderos
    (λ = 0,05, 1.500 observaciones), el 100 % de las ventanas tiene parámetros,
    hay CV(λ) en los 100 pares, la mediana de la vida media es 10,8 días frente a
    13,9 verdaderos (sesgo del AR(1) con 252 observaciones) y el ADF rechaza solo
    en el 48 % de las ventanas: por eso no puede usarse como filtro.

### E9 — Correlaciones por par sobre fechas comunes
- **Decisión:** los log-rendimientos de cada par se calculan sobre la intersección
  de fechas de sus precios; el número de observaciones se muestra por celda. Las
  fechas duplicadas siguen la política de la validación (E11).
- **Alternativas:** calcular rendimientos sobre el calendario propio de cada serie
  y luego alinear.
- **Justificación:** en un par cripto + bolsa, el rendimiento cripto de viernes a
  lunes debe incluir el fin de semana para ser comparable con el de la bolsa.
- **Fecha:** 2026-10-10

### E10 — Ventanas de estabilidad no evaluables (complementa E5)
- **Decisión:** una ventana donde Engle–Granger no se puede calcular (p. ej. una
  serie constante en la ventana) o es colineal (estadístico −∞) queda con
  `evaluable = False`, `passed = False` y su motivo: cuenta como fallida y la
  primera razón de inestabilidad nombra sus fechas (hasta 3, más «y N más»). El
  cambio de signo de β usa solo las ventanas con β. La exportación incluye
  `evaluable` y `motivo` por ventana.
- **Alternativas:** excluir la ventana del denominador; propagar el error (antes:
  todo el par pasaba a ⛔ `CALCULO_NO_POSIBLE` con un mensaje engañoso).
- **Justificación:** la estabilidad es un diagnóstico; una ventana no evaluable no
  invalida el Engle–Granger de toda la formación, y un p = 0 por colinealidad no
  es evidencia.
- **Fecha:** 2026-10-10

### E11 — Fechas duplicadas en las correlaciones (complementa E9)
- **Decisión:** `mrlab.stats.correlation` aplica la misma política que la
  validación (orden estable, conservar la última aparición de cada fecha y luego
  descartar NaN). Una entrada sin índice de fechas da `DataError` y una ventana
  móvil menor que 2 da `ConfigError VENTANA_INVALIDA`; nunca un `ValueError`
  crudo de pandas.
- **Alternativas:** lanzar `DataError` ante duplicados.
- **Justificación:** la caché conserva los duplicados crudos a propósito y la
  pestaña de correlaciones debe funcionar con la misma política que el resto del
  análisis.
- **Fecha:** 2026-10-10

### E12 — Coherencia de `AnalysisConfig`
- **Decisión:** `AnalysisConfig` lanza `ConfigError CONFIGURACION_ANALISIS_INVALIDA`
  en español (al construirse y al cargar un `run_config.json`) si:
  `stability_window < 2·(maxlag + 1) + 30` (56 con `maxlag = 12`; `2·(maxlag + 1)`
  es el mínimo técnico de `coint` y 30 el margen para que la prueba y el OU por
  ventana tengan sentido); `stability_step < 1`; `maxlag < 0`;
  `min_train_obs < max(30, 2·(maxlag + 2) + 1)`; `max_fail_frac` fuera de [0, 1];
  `half_life_min < 0` o `≥ half_life_max`; `correlation_min` fuera de [−1, 1];
  horizontes de pronóstico vacíos o < 1; `forecast_level` fuera de (0, 1);
  `fdr_method` distinto de `bh`/`by`. `alpha_level` y `fdr_q` conservan sus
  restricciones.
- **Alternativas:** dejar que fallen los cálculos (antes daba ⛔ «Datos
  insuficientes» o excepciones crudas).
- **Justificación:** un error de configuración no debe presentarse como falta de
  datos.
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
- **Decisión:** sin un OU válido no hay σ_eq ni z: la estrategia queda
  bloqueada. Si el OU es válido pero el par no es ✅, el backtest se permite con un
  banner rojo permanente «El par no superó los filtros: resultado solo
  ilustrativo».
- **Actualización (2026-10-10):** «OU válido» significa `0 < b < 1` **y** que la
  prueba de Dickey–Fuller aumentada (ADF, rezagos por AIC) rechace la raíz
  unitaria (O7); sobre el spread de formación equivale a que Engle–Granger SIN
  AJUSTAR de la dirección elegida rechace al nivel α. El banner
  también se muestra si el par es ✅ pero tiene alguna razón de descarte, y cita
  los motivos (O5a).
- **Alternativas:** bloquear siempre que el par no sea ✅.
- **Justificación:** ver fallar la estrategia en un par sin evidencia es
  pedagógicamente valioso, siempre que no pueda confundirse con evidencia.
- **Fecha:** 2026-10-10

### O5a — Resultado «solo ilustrativo» con sus motivos (complementa O5 y A1)
- **Decisión:** `illustrative_only` = estado distinto de ✅ **o** cualquier razón
  del panel «¿Por qué se descartó?» (vida media fuera de [1, 126], filtro de
  correlación, costes que superan el PnL bruto, un segmento que no se pudo
  simular, etc.). El banner «El par no superó los filtros: resultado solo
  ilustrativo.» se completa con los motivos (`PairAnalysis.illustrative_reasons`
  e `illustrative_banner`) y encabeza los avisos de cada backtest; el LEEME lo
  muestra una sola vez. Las razones que solo se conocen al simular
  (`COSTES_SUPERAN_BRUTO` en validación, `SIMULACION_NO_POSIBLE`) las añade el
  runner con `add_discard_reasons`, que mantiene esta regla y actualiza el banner
  de cada backtest ya simulado.
- **Alternativas:** depender solo del estado; documentar que la vida media no
  afecta al banner; dejar fuera del banner las razones que añade el runner.
- **Justificación:** un par ✅ con vida media < 1 día se presentaba sin banner
  mientras el panel de descartes decía «no operable», y un par con costes que
  superan el PnL bruto se exportaba con `solo_ilustrativo = no` aunque el panel lo
  descartaba. Eso contradice la sección 0: el resultado del backtest no puede
  contradecir el panel de descartes.
- **Fecha:** 2026-10-10

### O6 — Simulación: cruce de μ y vida media empírica
- **Decisión:** se exige X₀ ≠ μ; el cruce se detecta en tiempo discreto (sesgo
  hacia arriba, explicado); la vida media empírica es el primer h en que la media
  de las trayectorias cae a la mitad de (X₀ − μ), con interpolación lineal.
- **Alternativas:** puente browniano para detectar cruces entre observaciones.
- **Justificación:** simple y explicable; la diferencia entre mediana del tiempo
  de cruce y vida media es en sí una lección.
- **Fecha:** 2026-10-10

### O7 — Validez del OU: 0 < b < 1 y la prueba ADF rechaza la raíz unitaria
- **Decisión:** un ajuste OU solo es VÁLIDO si `0 < b̂ < 1` **y** una prueba de
  Dickey–Fuller AUMENTADA (ADF) rechaza la raíz unitaria al nivel α (0,05 por
  defecto; en el análisis del par, α = `alpha_level`): se rechaza si el p-value
  es ≤ α. El AR(1) por MCO (O1) solo estima a, b, λ, μ, σ, σ_eq y la vida media;
  la decisión la toma el ADF, con los rezagos de las diferencias elegidos por AIC
  hasta `maxlag` (`AnalysisConfig.maxlag` en el análisis, el mismo de
  Engle–Granger, E3; 12 por defecto en `fit_ou`).
  - Serie observada (`estimated_residual=False`, valor por defecto de `fit_ou`):
    `adfuller(x, regression="c", autolag="AIC")`, p-value de MacKinnon con N = 1.
  - Residuo de la regresión de cointegración estimada con los mismos datos
    (`estimated_residual=True`: el spread de formación del análisis del par):
    `adfuller(resid, regression="n", autolag="aic")` y p-value de MacKinnon N = 2
    con constante, que es exactamente el paso interno de `statsmodels.coint`. Por
    eso, en formación, OU válido ⟺ `0 < b̂ < 1` y p-value de Engle–Granger SIN
    AJUSTAR de la dirección elegida ≤ α (mismo estadístico, mismos rezagos y mismo
    p-value; una prueba lo verifica). Engle–Granger decide el estado con el p-value
    AJUSTADO (A8), que es mayor o igual: un par ✅ siempre rechaza aquí, y un par
    con p sin ajustar ≤ α < p ajustado tiene un OU válido pero queda ⚠️ (backtest
    solo ilustrativo, O5).
  - `df_crit` es el valor crítico que corresponde exactamente a ese p-value
    (inversión de `mackinnonp`), así que «estadístico ≤ valor crítico» y «p-value
    ≤ α» nunca se contradicen. La tabla de muestra finita (`mackinnoncrit`) difiere
    en menos de 0,01 con n ≥ 252 (al 5 % con 899 transiciones: −2,865 frente a
    −2,862 con N = 1; −3,343 frente a −3,336 con N = 2), pero haría discrepar la
    validez del OU y Engle–Granger en esa franja.
  - Si no se rechaza: estado `NON_STATIONARY`, `valid = False`,
    `near_unit_root = True` (significa «0 < b̂ < 1 pero b̂ no se distingue de 1»),
    sin λ, μ, σ, σ_eq, vida media, EE(μ̂) ni bandas. Se conservan a, b y la prueba
    (`df_stat`, `df_crit`, `df_pvalue`, `df_lags`, `df_alpha`,
    `df_estimated_residual`), y el mensaje en español la describe con sus rezagos.
    La razón de descarte es `OU_NO_ESTACIONARIO`, que bloquea la estrategia
    (sustituye a la antigua `RAIZ_UNITARIA_CERCANA`, que solo avisaba).
  - Si la prueba no se puede calcular, el ajuste es `INSUFFICIENT` (A9).
  - `fit_ou(require_unit_root_rejection=False)` calcula e informa la prueba igual,
    pero la validez solo exige `0 < b̂ < 1`. Lo usan las ventanas de estabilidad
    (E8), donde la prueba de cada ventana es un dato aparte con N = 1.
  - Sustituye la parte de A9 que decía que `near_unit_root` es solo un aviso y
    precisa O5.
- **Alternativas:** mantener solo `0 < b < 1` con un aviso (regla original);
  Dickey–Fuller SIN rezagos sobre el AR(1), con la tabla de muestra finita
  (primera versión de esta decisión); Phillips–Perron; usar siempre N = 1.
- **Justificación:** simulación con 900 observaciones (una formación de 1.500 con
  la división 60/20/20) y 2.000 repeticiones (MATEMATICAS §8.5):
  - Con la regla original, un paseo aleatorio observado tiene `0 < b̂ < 1` en el
    96,2 % de los casos (sesgo de Dickey–Fuller); la prueba lo acepta en el 5,3 %.
  - El residuo de MCO entre dos paseos independientes tiene `0 < b̂ < 1` en el
    99,1 %; la tabla N = 1 lo acepta en el 14,3 % y la N = 2 en el 5,4 % (≈
    nominal).
  - Sin rezagos, la prueba rechazaba de más cuando los incrementos tienen
    autocorrelación, como con ruido de microestructura: un paseo aleatorio más un
    ruido de medición iid de 0,5 % o 1 % salía «OU válido» en el 11,9 % y el
    38,0 % de los casos. Con el ADF, en el 5,2 % y el 5,7 %.
  - En el análisis completo de 200 pares de paseos independientes de 1.500
    observaciones: con la regla original los 200 daban un OU válido; con la
    primera versión de O7, 17; con la actual, 21 (10,5 %, ninguno ✅). Supera el
    5 % porque se elige la dirección de Engle–Granger con menor p-value, y en los
    200 casos la validez coincide con Engle–Granger sin ajustar.
  - Que el OU y Engle–Granger usen el mismo cálculo evita que el alumno vea dos
    veredictos distintos sobre la misma serie. La sección 0/17 prohíbe sugerir
    reversión sin evidencia, y la prueba obligatoria de la sección 14 («random
    walk → no estacionario») ya no depende de una semilla elegida.
- **Fecha:** 2026-10-10

### O8 — Series colineales y escala absoluta del spread
- **Decisión:** si la dirección elegida de Engle–Granger tiene estadístico −∞
  (series idénticas salvo escala), no se ajusta el OU (`OUFit` con estado
  «datos insuficientes» y la explicación de colinealidad) y el par se descarta con
  `SERIES_COLINEALES`, que bloquea la simulación. Además, `fit_ou` acepta `scale`
  (en el análisis, la desviación de f(y) en formación) y devuelve «datos
  insuficientes» si `std(spread) < 10⁻¹⁰·scale` (ruido de redondeo).
- **Alternativas:** solo la comprobación relativa existente (s² frente a la
  varianza de X); bloquear en la validación de datos.
- **Justificación:** el OU salía válido con σ_eq ≈ 10⁻¹⁶ y el backtest operaba
  ruido de redondeo. Nota: dos ETF casi idénticos (R² ≥ 1 − 1,5·10⁻⁶ en formación)
  también quedan bloqueados, con la explicación.
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
- **Nota:** el registro persistente en `data/studies/` llega en la fase 2; la
  versión mínima de la fase 1 es S10a.

### S10a — Prueba contaminada en la fase 1 (complementa S10)
- **Decisión:** `RunConfig.test_history` guarda los `config_hash` con los que se
  ha simulado la prueba (fuera del hash). Al simularla con hash h: contaminada =
  el historial contiene algún hash distinto de h. La marca es permanente: volver a
  la configuración original no la quita mientras el historial la conserve. h se
  añade si no estaba. Si la prueba no se simula: `test_contaminated = None` («no
  aplica») y el historial se conserva. El LEEME, `pruebas_estadisticas.csv`
  (sección `prueba`) y la demo dicen «SÍ», «no (según el historial registrado en
  run_config.json…)» o «no aplica», con una nota de alcance; en la demo, el título
  del bloque de prueba pasa a «Prueba (CONTAMINADA…)».
- **Alternativas:** no afirmar nada («no verificado») hasta la fase 2; la regla
  literal «h no está en el historial» (permitiría volver a la configuración
  original sin marca).
- **Justificación:** la exportación decía «Prueba contaminada: no.» sin
  verificarlo, aunque la prueba se hubiera re-ejecutado con otros umbrales. El
  historial viaja con la configuración y detecta el caso real reproducido con la
  demo (editar los umbrales del `run_config.json` y volver a ejecutar). Se declara
  su límite: si se borra el historial a mano, no se detecta.
- **Actualización (2026-10-10), límite de la fase 1:** tampoco se detecta si una
  ejecución nueva parte de una configuración sin ese historial: exportar en otra
  carpeta (`--output-dir`) o partir de un `run_config.json` copiado antes de
  ejecutar la prueba y exportar fuera de la carpeta original. La nota de alcance
  (`TEST_HISTORY_NOTE`, también en el `LEEME.txt` exportado) y el README lo dicen;
  la demo conserva el historial de la carpeta de destino (S10b). La solución
  completa es el registro persistente de estudios de la fase 2 (S10).
- **Fecha:** 2026-10-10

### S10b — Historial de la prueba en la demo (complementa S10a y A7)
- **Decisión:** antes de exportar, la demo fusiona el `test_history` del
  `run_config.json` (con o sin prefijo `SINTETICO_`) que ya haya en la carpeta de
  destino cuando es del mismo par (mismo proveedor y símbolo de A y B; los
  proxies ejecutables y las fechas no cuentan, para no perder historial). Lo hace
  en ejecuciones nuevas y también con `--reproducir`. El archivo existente se lee
  de forma tolerante, sin verificar el hash, porque `test_history` está fuera de
  él. Si la fusión añade hashes, `run_pair` se vuelve a llamar con el historial
  fusionado, para que `export_basename` siga siendo la única fuente del nombre de
  la carpeta y la marca se calcule con el historial completo. Un
  `run_config.json` ilegible da `DEMO_HISTORIAL_ILEGIBLE`; uno de otro par con
  historial, `DEMO_CARPETA_DE_OTRO_PAR` (código de salida 2, sin sobrescribir).
  Uno de otro par sin historial se reemplaza con aviso. La opción
  `--reiniciar-historial` descarta a conciencia el historial de la carpeta de
  destino y lo avisa.
- **Alternativas:** un registro mínimo por carpeta de salida
  (`historial_prueba.json`); solo avisar y no sobrescribir un historial no vacío;
  esperar al registro de la fase 2.
- **Justificación:** una ejecución normal de la demo creaba un `RunConfig` con el
  historial vacío y sobrescribía el `run_config.json` de la misma carpeta (la
  carpeta por defecto es la misma para el mismo par). Así, tras ver la prueba,
  bastaba volver a ejecutar la demo y editar los umbrales para que la prueba
  saliera «no contaminada». Leer el historial que ya está en la carpeta de destino
  cierra ese camino sin introducir un archivo nuevo.
- **Fecha:** 2026-10-10

### S11 — Avisos del motor: corto de cripto spot y capital agotado
- **Decisión:** (1) si alguna operación deja corta una pierna de clase cripto, el
  backtest avisa que el corto de cripto spot requiere derivados (perpetuos con
  funding), fuera de la v1, y que el préstamo cobrado es una aproximación; se
  nombran los símbolos y el aviso llega al LEEME. (2) El aviso «El capital del par
  se agotó durante el segmento» se da si cualquier cierre del diario tiene capital
  ≤ 0, y en el último cierre se registra también el evento `capital_agotado` (sin
  orden).
- **Alternativas:** (1) bloquear los cortos de cripto; (2) avisar solo cuando
  exista el evento.
- **Justificación:** secciones 2.2 y 7.4; antes se simulaban cortos de cripto sin
  ningún aviso, y el capital podía terminar negativo sin aviso si se agotaba en el
  último cierre.
- **Fecha:** 2026-10-10

### S12 — Precios ejecutables no positivos
- **Decisión:** si algún precio EJECUTABLE (la propia serie con spread en precios,
  o el proxy) es ≤ 0 en formación, validación o prueba, el par recibe la razón de
  descarte `PRECIO_EJECUTABLE_NO_POSITIVO`, con el instrumento, el tramo y la
  primera fecha: la simulación queda bloqueada y el análisis se muestra. Con
  proxy, esas fechas ya no se excluyen del análisis.
- **Alternativas:** capturar el error del motor por segmento (se mantiene como
  defensa adicional, A10); excluir esas fechas.
- **Justificación:** el dimensionamiento (cantidad = nocional / precio) y la
  valoración necesitan precios positivos; antes `run_pair` abortaba y se perdía
  todo el análisis (caso WTI de abril de 2020).
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

### C5 — Rangos razonables de costes
- **Decisión:** `CostParams` exige comisión y slippage en [0, 1.000] pb (hasta
  10 % por pierna), tasas anuales de préstamo y financiación en [0, 1] (fracción
  anual) y `day_count` en (0, 366]. Fuera de rango → `ConfigError COSTES_INVALIDOS`
  en español, con el campo, el rango y el valor recibido; si una tasa parece un
  porcentaje (1 < x ≤ 100), sugiere dividir por 100.
- **Alternativas:** solo `day_count > 0`; `day_count ∈ {360, 365}`.
- **Justificación:** un `run_config.json` editado a mano llegaba a una división por
  cero (`day_count = 0`) o convertía préstamo y financiación en ingresos
  (`day_count < 0`); escribir 3 en lugar de 0,03 es un error habitual.
- **Fecha:** 2026-10-10

## Ranking y cartera (fases 4 y 5)

### R1 — Puntaje compuesto
- **Decisión:** puntaje = Σ wᵢ·sᵢ con sᵢ ∈ [0, 1], fórmulas visibles:
  cointegración `clip(1 − p_aj/0,10)`; estabilidad `1 − fracción de ventanas
  fallidas` (0 si β cambia de signo); vida media 1 en [5, 63] días, lineal hasta 0
  en 1 y 126; calidad OU `½·min(1, p_LB/0,05) + ½·(1 − min(1, CV de λ))`, con
  el CV de λ de las ventanas con `0 < b̂ < 1` (E8; la prueba ADF por ventana no lo
  filtra); costes
  `clip(1 − coste ida y vuelta / ((z_entrada − z_salida)·σ_eq))`; validación
  `clip(Sharpe_validación/2)` (×½ si hay menos de 10 operaciones). Pesos por
  defecto 0,25 / 0,20 / 0,15 / 0,15 / 0,10 / 0,15. Los pares no ✅ o con OU
  inválido van a una tabla de descartados con sus causas. La prueba va en una
  columna aparte y no entra en el puntaje.
- **Alternativas:** ordenar por p-value; ordenar por PnL.
- **Justificación:** la especificación prohíbe ordenar por PnL retrospectivo.
  Calcular el CV de λ solo con las ventanas que rechazan la raíz unitaria lo
  sesgaría (solo quedan las de λ alto; ver E8).
- **Fecha:** 2026-10-10

### R2 — Walk-forward
- **Decisión:** la posición se cierra al final de cada ventana de prueba; si en una
  ventana el OU es inválido, no se opera en ella y se registra. Cada ventana de
  formación del walk-forward re-estima α y β, así que su OU sigue la regla
  completa de O7 (residuo estimado, N = 2: equivale a Engle–Granger sin ajustar en
  esa ventana). Con 504 observaciones la prueba tiene menos potencia que con toda
  la formación, y la interfaz lo explicará (a diferencia de las ventanas de
  estabilidad de E8, aquí sí se decide si se opera).
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
  `config_hash`, `code_version`, `series` ni `test_history` (I5a). La versión del
  código (hash de git, `-dirty` si hay cambios; variable `GIT_COMMIT` en Docker)
  se registra aparte.
- **Alternativas:** incluir la versión del código en el hash.
- **Justificación:** la misma configuración debe identificarse igual aunque cambie
  el código; la versión se audita por separado.
- **Fecha:** 2026-10-10

### I5a — Qué identifica la configuración (modifica I5)
- **Decisión:** `config_hash` excluye `config_hash`, `code_version`, `series` y
  `test_history` (`HASH_EXCLUDED`). `series` se excluye entero porque solo
  describe los datos (fecha de lectura, cobertura, notas) y las referencias
  fijadas por `snapshot_id` en `pair` ya identifican el contenido; `test_history`
  es un registro de ejecuciones, no un parámetro. `include_test` y `seeds` sí
  forman parte del hash (A3a, A3b). Versión de esquema 2: los archivos de la
  versión 1 se verifican con la definición de entonces (el JSON tal como se
  guardó, sin `config_hash` ni `code_version`) y se devuelven actualizados a la
  versión 2 con su hash nuevo; si no coinciden, se rechazan como cualquier
  archivo editado.
- **Alternativas:** excluir solo `retrieved_at`; no subir la versión (los
  archivos antiguos se rechazarían diciendo que «se modificaron»); aceptar los v1
  sin verificar.
- **Justificación:** los mismos CSV registrados en momentos distintos daban otro
  hash, y eso habría disparado en falso la marca de prueba contaminada (S10a).
  Subir la versión evita un mensaje engañoso con los archivos ya exportados.
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

### I7 — Settings como valores por defecto de los proveedores
- **Decisión:** `ProviderRegistry.from_toml` pasa a cada clase de proveedor que
  define `defaults_from_settings(settings)` esas opciones como valores por
  defecto; las claves de `providers.toml` mandan. `YahooProvider` toma
  `yahoo_max_retries` y `yahoo_backoff_base_seconds` de `mrlab.config.get_settings()`
  (variables `MRLAB_YAHOO_MAX_RETRIES` y `MRLAB_YAHOO_BACKOFF_BASE_SECONDS`) y
  valida esas opciones (entero ≥ 0; números ≥ 0 finitos) con `ConfigError` en
  español.
- **Alternativas:** leer Settings dentro de `YahooProvider`; un caso especial por
  nombre de proveedor en el registro.
- **Justificación:** mantiene el proveedor inyectable en pruebas y extensible a
  otros proveedores (p. ej. una clave de API); el TOML sigue siendo la
  configuración explícita. Antes `max_retries = -1` terminaba en un
  `AssertionError`.
- **Fecha:** 2026-10-10

### I8 — Metadatos del paquete dentro de `backend/`
- **Decisión:** `backend/tests/unit/test_packaging.py` lee `backend/pyproject.toml`
  con `tomllib` y comprueba que `project.readme`, `license`, `license-files` y
  todas las rutas de `[tool.setuptools]` (`where`, `package-dir`, `data-files`,
  `dynamic.*.file`…) quedan dentro de `backend/`: sin rutas absolutas ni «..».
  Incluye una autoverificación con la configuración que rompió la instalación
  (`readme = "../README.md"`).
- **Alternativas:** un paso de CI con `pip install --dry-run -e ./backend` en un
  clon limpio.
- **Justificación:** es la regresión del fallo de `pip install -e "backend[dev]"`
  en un clon nuevo (corregido en el commit ac0c276): sin prueba, la ruta podía
  volver sin que la suite lo detectara. La prueba no necesita red ni un clon.
- **Fecha:** 2026-10-10

## Integración de la fase 1

### A1 — Causas del estado y razones de descarte
- **Decisión:** `causes` contiene TODAS las causas activas del estado principal,
  ordenadas por la precedencia E6; si no hay ninguna, el estado es ✅ y lleva una
  única causa con el p-value ajustado de Engle–Granger. La validez del OU no es
  un estado: un OU no válido (O7: b fuera de (0, 1) o b que no se distingue de 1
  según la prueba ADF, `OU_NO_ESTACIONARIO`) aparece en «¿Por qué se descartó?»
  sin cambiar el estado. Lo mismo para la vida media fuera de [1, 126], el filtro
  de correlación, «costes superan el PnL bruto» en validación y un segmento que no
  se pudo simular (A10). Bloquean la simulación, sin cambiar el estado, las
  razones de `STRATEGY_BLOCKING_CODES`: OU no válido (`OU_NO_ESTACIONARIO`,
  `OU_B_NO_POSITIVO`, `OU_DATOS_INSUFICIENTES`), `SERIES_COLINEALES`,
  `PROXY_REQUERIDO`, `PROXY_NO_EJECUTABLE`, `PRECIO_EJECUTABLE_NO_POSITIVO` y
  `EJECUTABLES_IDENTICOS` (A2a).
  `strategy_allowed` = no ⛔, OU válido y ninguna de esas razones. Cualquier razón
  de descarte hace el resultado solo ilustrativo (O5a). Un error de datos durante
  los cálculos (p. ej. una serie constante en formación) da ⛔ con la causa
  `CALCULO_NO_POSIBLE`; una ventana de estabilidad no evaluable no (E10).
- **Alternativas:** convertir el OU no válido (o las razones que bloquean la
  simulación) en estados; mostrar solo la causa que decide el estado.
- **Justificación:** la especificación fija exactamente cinco estados y pide
  listar todas las causas; separar estado (evidencia estadística) y descartes
  (aptitud para operar) es más claro para el alumno.
- **Fecha:** 2026-10-10

### A2 — Proxy ejecutable: requerido, alineado por intersección y validado (actualizada)
- **Decisión:** con `exec_a`/`exec_b`, el análisis y el backtest usan solo las
  fechas en que cotizan también los instrumentos ejecutables (nunca se rellena) y
  se avisa cuántas fechas se excluyeron y del riesgo de base. El proxy debe estar
  en la misma moneda que la serie analizada (si no, ⛔; si su moneda es
  desconocida, ⛔ `MONEDA_DESCONOCIDA`). Además (2026-10-10):
  (a) una pierna no ejecutable (índice o futuro) sin proxy recibe la razón de
  descarte `PROXY_REQUERIDO`, que sugiere el proxy sin sustituirlo y bloquea la
  simulación; el análisis estadístico se muestra completo. Si el instrumento
  elegido como ejecutable tampoco es ejecutable: `PROXY_NO_EJECUTABLE`. «No
  ejecutable» significa `not AssetInfo.executable` **o** clase de activo en
  `NON_EXECUTABLE_CLASSES` (índice, futuro), tanto para la serie analizada como
  para el proxy: es una defensa además del validador de `AssetInfo`, que
  `model_copy` no ejecuta. `PROXY_REQUERIDO` nunca sugiere un proxy que coincida
  con la serie o el ejecutable de la otra pierna: lo dice y pide otro instrumento
  (A2a).
  (b) Las incidencias de la limpieza del proxy (duplicados, faltantes, saltos,
  huecos, precios ≤ 0) se añaden al informe de validación con su símbolo y la
  etiqueta «Instrumento ejecutable», y el resumen se recalcula con las fechas
  realmente usadas (D15a).
  (c) Las fechas con precio del proxy ≤ 0 ya no se excluyen: bloquean la
  simulación (S12).
  (d) El calendario se decide con todos los instrumentos (D17).
- **Alternativas:** operar el índice sobre su propia serie con un aviso (versión
  anterior de esta decisión); reindexar el proxy y rellenar huecos; excluir las
  fechas con precio ≤ 0; decidir «no ejecutable» solo con `AssetInfo.executable`
  (un índice construido sin el validador se simulaba sin proxy).
- **Justificación:** D1 confirmada exige ejecutar con un proxy elegido
  explícitamente: un índice no se puede comprar ni vender, así que simularlo como
  si se pudiera presentaba un PnL imposible. Las incidencias del proxy alimentan el
  PnL y deben verse. Coherente con D6 (nunca rellenar).
- **Fecha:** 2026-10-10

### A2a — Ejecutables idénticos (complementa A1 y A2)
- **Decisión:** nueva razón de descarte `EJECUTABLES_IDENTICOS`, incluida en
  `STRATEGY_BLOCKING_CODES`. Se activa si, tras resolver `exec_a`/`exec_b`, las dos
  piernas se operarían con el mismo instrumento (mismo proveedor y símbolo, sin
  distinguir mayúsculas ni espacios) o con precios proporcionales en las fechas
  usadas (desviación estándar de `ln(Pa/Pb)` < `10⁻¹⁰·máx(desv. ln Pa, desv. ln Pb)`,
  la misma tolerancia que O8). Solo se evalúa si hay algún ejecutable: sin
  ejecutables ya lo cubren `MISMO_ACTIVO` (D8) y `SERIES_COLINEALES` (O8). El
  estado del par no cambia y el análisis estadístico se muestra. Además,
  `PROXY_REQUERIDO` nunca sugiere como proxy el símbolo de la serie o del
  ejecutable de la otra pierna, y el error `DEMO_SIMBOLO_REPETIDO` de la demo ya
  no invita solo a renombrar: explica que el mismo instrumento no puede usarse en
  las dos piernas.
- **Alternativas:** comparar solo las series de análisis (versión anterior); un
  estado ⛔ (el análisis índice frente a ETF es legítimo: lo que no tiene sentido
  es la simulación).
- **Justificación:** `^GSPC`/`SPY` con `exec_a = SPY` salía ✅ con la estrategia
  permitida y operaba SPY contra sí mismo (el PnL eran solo costes), y el propio
  mensaje de `PROXY_REQUERIDO` sugería `SPY`. En la demo, una copia del mismo CSV
  con otro nombre pasaba sin aviso. Es la misma clase de error que O8, en la capa
  ejecutable.
- **Fecha:** 2026-10-10

### A3 — `run_config.json` autosuficiente
- **Decisión:** `run_pair` devuelve el RunConfig completado: cada referencia
  queda fijada a la instantánea usada (`snapshot_id`, conservando las fechas
  pedidas), `frozen_split`, `series`, `data_origin` según los datos (SINTETICO si
  alguna serie lo es, aunque la configuración dijera otra cosa), `include_test`
  (A3a), `seeds` (A3b), `test_history` (S10a), `code_version` y `config_hash`. Sin `snapshot_id` la referencia debe traer `start` y `end`. Una
  instantánea cuyo proveedor o símbolo no coincide con la referencia se rechaza.
  Un `run_config.json` escrito a mano sin `config_hash` se acepta y se le calcula;
  uno con hash que no coincide se rechaza.
- **Alternativas:** guardar solo la configuración de entrada y resolver los
  datos en cada ejecución.
- **Justificación:** guardar, cargar y volver a ejecutar debe dar los mismos
  bytes sin tocar el proveedor (D7, sección 15).
- **Fecha:** 2026-10-10

### A3a — `include_test` en el RunConfig
- **Decisión:** `RunConfig.include_test` forma parte del hash. En
  `run_pair(config, data, include_test=None)` el argumento solo puede ACTIVAR la
  prueba; con None o False manda la configuración. El valor efectivo queda en el
  RunConfig devuelto.
- **Alternativas:** que un False explícito desactive la prueba aunque la
  configuración la incluya; guardar `include_test` fuera del hash.
- **Justificación:** reproducir un `run_config.json` que incluía la prueba debe dar
  los mismos bytes aunque se omita `--incluir-prueba` (la orden «Para reproducir»
  de la demo no la lleva). Ejecutar o no la prueba cambia el resultado, así que
  debe cambiar la identidad.
- **Fecha:** 2026-10-10

### A3b — Semillas en `run_config.json`
- **Decisión:** `run_pair` rellena `RunConfig.seeds` con las semillas de la
  configuración más `{símbolo: SeriesMeta.seed}` de cada serie generada (el valor
  de la serie prevalece en la misma clave; claves ordenadas); forma parte del
  hash. El proveedor sintético y el CSV (campo `seed` del `.meta.json`) rellenan
  `SeriesMeta.seed`.
- **Alternativas:** reemplazar en lugar de fusionar; dejar la semilla solo en la
  nota de la fuente.
- **Justificación:** la sección 15 pide registrar las semillas (antes `seeds`
  estaba siempre vacío). Fusionar conserva semillas de otros usos (p. ej.
  bootstrap o simulación en fases futuras).
- **Fecha:** 2026-10-10

### A3c — Errores de formato de `run_config.json` en español
- **Decisión:** `load_run_config` traduce los tipos de error de pydantic más
  comunes a texto propio: «<ruta.del.campo>» <problema> (recibido: <valor>),
  hasta 5 y «y N problema(s) más». `details["errores"]` lleva lo mismo
  estructurado. Los errores de coherencia de los validadores
  (`UMBRALES_INCONSISTENTES`, `COSTES_INVALIDOS`, `CONFIGURACION_ANALISIS_INVALIDA`)
  conservan su código y nombran el archivo.
- **Alternativas:** mostrar solo la ruta del campo, sin el problema.
- **Justificación:** el usuario no debe ver textos de pydantic en inglés; la ruta
  y el valor bastan para corregir el archivo.
- **Fecha:** 2026-10-10

### A4 — Formato de la exportación
- **Decisión:** `tipo_datos` es la primera columna de cada CSV. Con datos
  sintéticos TODOS los archivos (también `run_config.json` y `LEEME.txt`) y la
  carpeta sugerida llevan el prefijo `SINTETICO_`. `pruebas_estadisticas.csv` va
  en formato largo (`seccion, elemento, campo, valor`) e incluye estado, causas,
  descartes y métricas por segmento. Booleanos «sí»/«no»; fracciones como
  fracción; `-0` se escribe `0`. La opción «CSV para Excel en español» añade el
  BOM de UTF-8. El ZIP fija fecha, sistema de origen y permisos de cada entrada.
  Todos los campos llevan nombre en español, también los nuevos del OU
  (`valor_critico_dickey_fuller`, `p_valor_dickey_fuller`,
  `rezagos_dickey_fuller`, `nivel_dickey_fuller`,
  `valores_criticos_residuo_estimado`); las ventanas de estabilidad de
  Engle–Granger incluyen `evaluable` y `motivo` (E10); las del OU, `estimable`,
  `b` y la prueba ADF de la ventana (`rechaza_raiz_unitaria`,
  `estadistico_dickey_fuller`, `p_valor_dickey_fuller`, `rezagos_dickey_fuller`),
  y su resumen `ventanas_sin_parametros`, `cv_lambda`, `nivel_dickey_fuller`,
  `ventanas_con_prueba_dickey_fuller` y `ventanas_que_rechazan_raiz_unitaria`
  (E8). La sección `prueba` dice si la prueba se pidió, si se ejecutó y si está
  contaminada (S10a).
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

### A7 — Demo reproducible (actualizada)
- **Decisión:** `examples/demo_ko_pep.py` lee los metadatos de `<archivo>.meta.json`
  si existe (lo crean `make fetch-example` y el generador sintético); si no, exige
  `--currency`, `--asset-class` y `--price-type` (D4). Usa la caché `data/cache`
  (o `--cache-dir`), ejecuta la prueba solo con `--incluir-prueba` y admite
  `--reproducir <run_config.json>` (solo caché, nunca descarga). Además
  (2026-10-10):
  - `--exec-a`/`--exec-b`: CSV del instrumento ejecutable de cada pierna (índices
    y futuros), con su `.meta.json` o con `--asset-class-exec-a`, etc. Los
    ejecutables heredan la moneda y el tipo de precio generales, pero su clase de
    activo debe declararse aparte (suele diferir: índice frente a ETF). Sin proxy,
    la demo muestra el análisis y explica el bloqueo `PROXY_REQUERIDO`.
  - Metadatos con `parse_metadata`/`build_metadata`; registra cada CSV con su
    nombre original (D10) y avisa si los datos se tratan como SINTÉTICOS (por la
    columna `tipo_datos` o por el prefijo `SINTETICO_`, detectado con
    `origin_from_filename`).
  - Errores siempre en español y sin trazas de Python (código de salida 2),
    también los de argparse (traducidos) y los de archivos.
  - `--reproducir` respeta `include_test` del `run_config.json` y no admite
    `--csv-*`, `--exec-*`, `--start` ni `--end`.
  - Orden «Para reproducir»: usa el intérprete actual (`sys.executable`) y rutas
    absolutas: el script, `--reproducir`, `--cache-dir` (si se dio o si no es el
    predeterminado de Settings) y `--output-dir` (si no es `examples/output`), más
    `--excel-es` si se usó. Cita con `subprocess.list2cmdline` en Windows
    (`os.name == "nt"`), con una pista para PowerShell («& » delante), y con
    `shlex.quote` en POSIX. Funciona desde cualquier carpeta; tal cual reescribe la
    misma carpeta con los mismos bytes, y para comparar se añade al final
    `--output-dir <otra carpeta>` (comprobado por pruebas, también con la prueba
    incluida, con proxy y ejecutando la orden desde otra carpeta).
  - Historial de la prueba: conserva el de la carpeta de destino y admite
    `--reiniciar-historial` (S10b).
  - argparse en español en los tres scripts de `examples/`: módulo común
    `examples/_cli.py` (`SpanishArgumentParser` con `-h` en español, «uso:»,
    «opciones:», «argumentos posicionales:» y la traducción de los errores,
    incluida «ignored explicit argument» → «esta opción no admite un valor; se
    recibió …» antes del reemplazo genérico; más `iso_date` y `quote_command`). Los
    scripts añaden `examples/` a `sys.path` para funcionar también con `python -P`.
    `generar_csv_sinteticos.py` acepta un argumento posicional opcional «carpeta»,
    `-h/--help` no escribe nada y rechaza destinos que empiezan por «-» (sugiere
    `./-nombre`). `fetch_ko_pep.py` rechaza `--end` anterior a `--start` antes de
    cualquier descarga.
  - Muestra la marca «Prueba contaminada» con el texto de la exportación (S10a), el
    banner «solo ilustrativo» con sus motivos una sola vez (O5a) y, en el OU, la
    prueba de Dickey–Fuller aumentada con sus rezagos, su p-value y su valor
    crítico (O7).
- **Alternativas:** caché temporal en cada ejecución; heredar también la clase de
  activo en los ejecutables; imprimir la orden de reproducción sin las opciones de
  caché y formato, o con `python` y rutas relativas a la raíz del repositorio
  (versión anterior); leer `sys.argv` a mano en el generador.
- **Justificación:** con una caché persistente el `run_config.json` exportado se
  puede reproducir más tarde; la orden impresa debe reproducir exactamente lo
  mismo (antes, con `--excel-es` o una caché propia, no lo hacía; ejecutada desde
  otra carpeta o en `cmd.exe` con una ruta con espacios, fallaba). El generador
  tomaba `--help` como carpeta de destino y escribía archivos en ella, y la
  descarga mostraba su ayuda y sus errores en inglés.
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
    primer intento: con 4, hasta 5 llamadas y esperas de 1, 2, 4 y 8 s (D3). Los
    avisos de logging de yfinance se silencian (D16); los valores por defecto
    salen de Settings (I7).
  - Caché: el hash de la instantánea combina el contenido con la identidad de la
    serie (activo, moneda, tipo de precio, intervalo y rango pedido); la fecha de
    descarga no entra en el hash. El id se verifica al leer (D12).
  - CSV: se relee en cada carga (el usuario puede subir un archivo corregido con
    el mismo símbolo; cada versión queda como instantánea propia); si no es UTF-8
    válido se interpreta como Windows-1252 (Excel en español). La columna
    opcional `tipo_datos` y el prefijo `SINTETICO_` del nombre marcan los datos
    como sintéticos (D10).
  - OU: `near_unit_root` (b̂ entre 0 y 1 pero indistinguible de 1 según la
    prueba ADF) hace el ajuste NO válido (O7; antes era solo un aviso). Spread
    constante, determinista o nulo frente a la escala de las series →
    `INSUFFICIENT` (O8). Con NaN en el spread, el AR(1) solo usa pares
    consecutivos finitos y la prueba ADF se aplica al tramo continuo sin NaN más
    largo (nunca une tramos). Si ese tramo tiene menos de 30 puntos, si la
    regresión ADF es de rango incompleto (p. ej. ruido de redondeo de ~10⁻¹⁶) o si
    el estadístico no es finito, la prueba no es calculable y el ajuste es
    `INSUFFICIENT`. En series cortas, `maxlag` se reduce al máximo que admite
    `adfuller`. Ljung–Box con `model_df=1`. Simulación con horizonte máximo de
    2.520 pasos.
  - Backtest: salida forzada (`ExitReason.FORCED`, máxima prioridad, sin
    enfriamiento ni rearme) si el capital se agota con posición abierta o si una
    reducción de apalancamiento con unidades enteras dejaría una pierna en 0. La
    reducción por apalancamiento solo se dispara si el límite se supera en más
    de un 1 %.
- **Alternativas:** documentarlas solo en los docstrings.
- **Justificación:** son decisiones no especificadas que afectan resultados o
  mensajes; deben quedar auditables en un único lugar.
- **Fecha:** 2026-10-10

### A10 — Un segmento que no se puede simular no tumba el análisis
- **Decisión:** si el motor rechaza un segmento con un `LabError`, `run_pair` no
  simula ese segmento. La causa va a `discard_reasons` (`SIMULACION_NO_POSIBLE`),
  a `RunResult.segment_errors` y a la sección SEGMENTOS del LEEME; los demás
  segmentos y el análisis se conservan, y el resultado pasa a ser solo
  ilustrativo (O5a). Los errores que no son `LabError` (defectos de programación)
  se propagan.
- **Alternativas:** abortar toda la ejecución.
- **Justificación:** defensa en profundidad del hallazgo «spread en precios con
  precio ≤ 0» (el análisis además bloquea la estrategia con precios ejecutables
  ≤ 0, S12); perder la validación y la exportación por un segmento es
  desproporcionado.
- **Fecha:** 2026-10-10

### A11 — Integración de las correcciones de la revisión
- **Decisión:** las correcciones de la revisión adversarial de la fase 1 se
  integraron con estas reglas de convivencia: (1) las razones de descarte que el
  runner conoce al simular se registran con `add_discard_reasons` para que
  `illustrative_only`, el banner y la exportación no se contradigan (O5a); (2) la
  exportación traduce todos los campos nuevos (A4); (3) los ejemplos usan las
  funciones nuevas del núcleo (`parse_metadata`, `build_metadata`,
  `origin_from_filename`, `register(..., filename=...)`) y el generador escribe
  `tipo_datos` y `seed` (D9, D10).
- **Alternativas:** dejar cada corrección aislada en su módulo.
- **Justificación:** varias correcciones cambiaban contratos compartidos (OU,
  descartes, RunConfig); sin integrarlas, la exportación decía
  `solo_ilustrativo = no` con razones de descarte y exportaba campos en inglés.
- **Fecha:** 2026-10-10

### A12 — Integración de la segunda ronda de correcciones
- **Decisión:** al integrar la segunda ronda (O7 con ADF aumentado, E8, A2a, D15a,
  S10b, A7) se aplicaron las mismas reglas de convivencia que en A11: la
  exportación traduce todos los campos nuevos (A4), la demo muestra la prueba ADF
  con sus rezagos y su p-value, y las pruebas de otros módulos que dependían de la
  regla anterior se actualizaron con escenarios que siguen probando lo mismo (p.
  ej. «par sin evidencia con OU válido» pasa a usar una relación con ruptura cuyo
  Engle–Granger sin ajustar rechaza: con O7, un OU válido sobre el spread de
  formación exige ese rechazo).
- **Alternativas:** mantener la prueba del runner con su escenario anterior
  (p sin ajustar 0,067: ahora su OU no es válido y la prueba ya no comprobaba lo
  que dice).
- **Justificación:** las correcciones cambiaban contratos compartidos (`OUFit`,
  `OUWindow`, `OUStability`, `STRATEGY_BLOCKING_CODES`, el informe de validación).
- **Fecha:** 2026-10-10
