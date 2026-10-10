# Matemáticas del laboratorio

> **Herramienta educativa. No constituye asesoría de inversión. Resultados
> históricos y simulados no garantizan resultados futuros.**

Este documento reúne todas las fórmulas que usa el Laboratorio de Reversión a la
Media. Está escrito para quien sabe operar pero no domina la econometría. Cada
tema sigue el mismo orden:

1. **Intuición:** qué problema resuelve y por qué importa.
2. **Fórmula:** cómo se calcula exactamente en el laboratorio.
3. **Ejemplo numérico**, cuando ayuda.
4. **En el código:** el módulo de `backend/src/mrlab/` que lo implementa.

Las decisiones citadas entre paréntesis (E2, O2, S1, …) están explicadas en
[`DECISIONS.md`](DECISIONS.md). Los requisitos originales están en
[`ESPECIFICACION.md`](ESPECIFICACION.md).

## Contenido

0. [Notación y unidades](#0-notación-y-unidades)
1. [Rendimientos logarítmicos](#1-rendimientos-logarítmicos)
2. [Correlación, cointegración y aptitud para OU](#2-correlación-cointegración-y-aptitud-para-ou)
3. [Series I(0) e I(1): pruebas ADF y KPSS](#3-series-i0-e-i1-pruebas-adf-y-kpss)
4. [Cointegración: prueba de Engle–Granger](#4-cointegración-prueba-de-englegranger)
5. [Pruebas múltiples: Benjamini–Hochberg](#5-pruebas-múltiples-benjaminihochberg)
6. [Estabilidad entre ventanas y estado del par](#6-estabilidad-entre-ventanas-y-estado-del-par)
7. [El spread y el hedge ratio](#7-el-spread-y-el-hedge-ratio)
8. [El proceso de Ornstein–Uhlenbeck](#8-el-proceso-de-ornsteinuhlenbeck)
9. [Bandas, intervalos e incertidumbre de parámetros](#9-bandas-intervalos-e-incertidumbre-de-parámetros)
10. [z-score fijo y z-score móvil](#10-z-score-fijo-y-z-score-móvil)
11. [Reglas de la estrategia](#11-reglas-de-la-estrategia)
12. [Dimensionamiento de las piernas](#12-dimensionamiento-de-las-piernas)
13. [División de la muestra y convención de ejecución](#13-división-de-la-muestra-y-convención-de-ejecución)
14. [PnL: ganancias y pérdidas en dinero](#14-pnl-ganancias-y-pérdidas-en-dinero)
15. [Costes](#15-costes)
16. [Métricas](#16-métricas)
17. [Mapa de fórmulas y código](#17-mapa-de-fórmulas-y-código)
18. [Referencias](#18-referencias)

---

## 0. Notación y unidades

| Símbolo | Significado | Unidades |
|---|---|---|
| `P_A,t`, `P_B,t` | Precio de cierre (ajustado) de A y de B en la observación `t` | moneda (p. ej. USD) |
| `ln` | Logaritmo natural (base e). En este documento `log` también significa `ln` | — |
| `r_t` | Rendimiento logarítmico | log-puntos por observación |
| `X_t` | Spread | log-puntos (spread log) o moneda (spread en precios) |
| `α`, `β` | Constante y hedge ratio del spread | según el tipo de spread |
| `λ` (≡ `θ`) | Velocidad de reversión del OU | 1/día de negociación |
| `μ` | Referencia estadística del modelo | unidades del spread |
| `σ` | Intensidad del ruido del OU | unidades del spread / √día de negociación |
| `σ_eq` | Desviación estándar estacionaria del spread | unidades del spread |
| `Δt` | Tiempo entre observaciones. En la v1, `Δt = 1` | día de negociación |
| `z_t` | z-score del spread | adimensional (cuántas `σ_eq`) |
| `q_A,t`, `q_B,t` | Cantidades con signo (+ largo, − corto) | unidades del activo (acciones, monedas) |
| `N` | Nocional de la pierna A | moneda |
| `G` | Exposición bruta objetivo | moneda |
| `C_t` | Costes de la observación `t` | moneda |

Convenciones:

- **Coma decimal y punto de miles**, como en la interfaz: `0,05`, `10.000 USD`.
- **Un log-punto** es una diferencia de logaritmos. Para movimientos pequeños,
  `0,01` log-puntos ≈ 1 % (decisión O4).
- **Un día de negociación** es una observación (decisión O4). Para pares de dos
  criptomonedas, el calendario es de 7 días y la unidad se llama «día de
  negociación (calendario de 7 días)». El calendario se decide con **todos** los
  instrumentos usados, también los ejecutables: un par de dos criptomonedas que se
  opera con un proxy de bolsa solo tiene días hábiles, así que es mixto (252
  periodos y «día de negociación»; decisión D17).
- **pb** = punto básico = 0,01 % = 1/10.000.

---

## 1. Rendimientos logarítmicos

### Intuición

Para saber si dos activos se mueven juntos día a día se comparan sus
**rendimientos**, no sus precios. Los precios tienen tendencia, y dos series con
tendencia parecen «relacionadas» aunque no tengan nada que ver (sección 7.4). El
rendimiento logarítmico tiene dos ventajas prácticas: se **suma en el tiempo** y
es **simétrico** (subir y volver a bajar al mismo precio suma cero).

### Fórmula

```text
r_t = ln(P_t / P_{t−1}) = ln P_t − ln P_{t−1}

Rendimiento simple:  R_t = P_t / P_{t−1} − 1
Relación:            r_t = ln(1 + R_t) ≈ R_t   (si R_t es pequeño)
Suma en el tiempo:   r_{t+1} + r_{t+2} + … + r_{t+k} = ln(P_{t+k} / P_t)
```

### Ejemplo

Un precio pasa de 100 a 110 y vuelve a 100.

| Tramo | Simple | Logarítmico |
|---|---|---|
| 100 → 110 | +10,00 % | +0,0953 |
| 110 → 100 | −9,09 % | −0,0953 |
| Suma | +0,91 % (engañoso: el precio no cambió) | 0 (correcto) |

**Cuidado:** los rendimientos logarítmicos se suman en el **tiempo**, pero **no
entre activos**. El rendimiento de una cartera es el promedio ponderado de los
rendimientos **simples**, no de los logarítmicos. Por eso el PnL en dinero nunca
se calcula a partir del spread logarítmico (sección 14).

### Política de fechas

- Los rendimientos de cada par se calculan sobre la **intersección** de fechas de
  sus precios (decisión E9). En un par cripto + bolsa, el rendimiento cripto de
  viernes a lunes incluye el fin de semana, igual que el de la acción.
- Una fecha sin precio no es una observación: se descarta y el rendimiento
  siguiente abarca el hueco. **Nunca** se rellena con el precio anterior.
- Los precios deben ser estrictamente positivos (el logaritmo de un número ≤ 0 no
  existe).

**En el código:** `mrlab.stats.correlation.log_returns`.

---

## 2. Correlación, cointegración y aptitud para OU

Son tres preguntas distintas. El laboratorio las muestra por separado porque
confundirlas es el error más común en pairs trading.

| Concepto | Pregunta | Se mide sobre | Herramienta |
|---|---|---|---|
| **Correlación** | ¿Se mueven juntos día a día? | Rendimientos | Pearson, Spearman |
| **Cointegración** | ¿Existe una combinación de sus log-precios que vuelve a un nivel estable? | Niveles (log-precios) | Engle–Granger (sección 4) |
| **Aptitud para OU** | ¿Ese spread revierte a una velocidad medible y útil para operar? | Spread | AR(1)/OU, vida media, diagnóstico (sección 8) |

### 2.1 Correlación de Pearson, de Spearman y móvil

```text
Pearson:   ρ = Σ (r_A,t − r̄_A)·(r_B,t − r̄_B) / sqrt( Σ (r_A,t − r̄_A)² · Σ (r_B,t − r̄_B)² )
Spearman:  la misma fórmula aplicada a los RANGOS (posiciones ordenadas) de los rendimientos
```

- **Pearson** mide relación lineal. Unos pocos días extremos pueden dominarla.
- **Spearman** mide relación monótona (si uno sube, el otro tiende a subir) y es
  menos sensible a los días extremos.
- Cada celda de la matriz muestra también `n`, el número de rendimientos
  comunes usados.
- **Correlación móvil:** Pearson de los últimos 63 rendimientos (por defecto),
  hasta `t` inclusive. Vale NaN mientras la ventana no está completa.
- La correlación es un **filtro exploratorio opcional** (desactivado por
  defecto). **Nunca** es una prueba de cointegración.

### 2.2 Muy correlacionados y no cointegrados (y al revés)

**Caso 1: correlación alta, sin cointegración** (escenario sintético
`SINT-CNC`). Dos paseos aleatorios cuyos movimientos diarios tienen correlación
`ρ = 0,9`, cada uno con volatilidad diaria del 1 %. La correlación de sus
rendimientos es ≈ 0,9. Pero observa el spread `ln A − ln B`:

```text
Var(cambio diario del spread) = 0,01² + 0,01² − 2·0,9·0,01·0,01 = 0,00002
Desviación diaria             = sqrt(0,00002) ≈ 0,45 %
```

Ese spread **también es un paseo aleatorio**: no tiene un nivel al que volver y
su dispersión crece con la raíz del tiempo.

| Días transcurridos | Desviación típica acumulada del spread |
|---|---|
| 1 | 0,45 % |
| 250 | 7,1 % |
| 1.000 | 14,1 % |

La correlación mide si los **cambios diarios** van en la misma dirección. No dice
nada sobre si las **diferencias acumuladas** se corrigen. Dos activos pueden
moverse juntos el 90 % de los días y aun así separarse sin límite.

**Caso 2: cointegración con correlación baja** (escenario `SINT-CBC`).
`ln A = α + β·ln B + X`, con B paseo aleatorio de volatilidad diaria 1 % y `X`
un OU con `λ = 0,5`, `σ = 0,03`, `β = 1`. El rendimiento de A es
`r_A = β·r_B + ΔX`, y el cambio diario del OU es grande comparado con `r_B`:

```text
σ_eq²            = σ²/(2λ) = 0,03²/1 = 0,0009        (σ_eq = 3 %)
Var(ΔX)          = 2·σ_eq²·(1 − e^{−λ}) = 2·0,0009·(1 − e^{−0,5}) ≈ 0,000708
Desviación de ΔX ≈ 2,66 % diario
corr(r_A, r_B)   = 0,01 / sqrt(0,01² + 0,000708) ≈ 0,35
```

Correlación de solo 0,35, y sin embargo el spread **siempre vuelve**: es un OU con
vida media `ln 2 / 0,5 ≈ 1,39` días y `σ_eq = 3 %`.

Con la semilla del preset (7) el laboratorio lo detecta así: A y B compatibles
con I(1) (ADF en niveles p ≈ 0,41 y 0,38; KPSS en niveles p < 0,01), Engle–Granger
con p ≈ 5·10⁻²⁷ (ajustado ≈ 8·10⁻²⁷), correlación de los rendimientos en
formación 0,33, estado ✅ y un OU válido con `λ̂ ≈ 0,53` (vida media estimada
1,31 días, `σ̂_eq ≈ 2,9 %`). Con otras semillas el resultado varía: en las
semillas 1 a 30, 22 dan ✅ y todas tienen correlación menor que 0,5 (media 0,35,
máximo 0,42). Casi todas las demás fallan porque el paseo aleatorio puro B sale
«ambiguo» con ADF + KPSS (le pasa a ≈ 10 % de los paseos aleatorios con la regla
E2 al 5 %), algo que ningún parámetro del OU corrige.

¿Por qué un OU tan rápido? Con un OU lento y poco ruidoso en B (versión anterior:
volatilidad de B 0,4 %, `λ = 0,1`, `σ = 0,015`), el OU dominaba también los
niveles de A, que parecía estacionaria: el par salía «⚠️ Supuestos dudosos» en la
mayoría de las semillas. Un OU rápido domina el cambio **diario** (correlación
baja) sin dominar el nivel acumulado de A (que sigue siendo I(1)). La vida media
resultante, cercana a 1 día, está en el límite inferior de lo operable con datos
diarios: el ejemplo enseña que correlación y cointegración son cosas distintas,
no un par para operar.

### 2.3 Aptitud para OU

Que un par esté cointegrado dice que el spread es estacionario, pero no dice
**a qué velocidad** revierte. Un spread con vida media de 300 días está
cointegrado y no sirve para operar en un horizonte de semanas. Un spread con vida
media menor que un día no se puede operar con datos diarios. La aptitud para OU se
evalúa con:

- ajuste OU válido: `0 < b < 1` **y** la prueba de Dickey–Fuller aumentada (ADF)
  rechaza la raíz unitaria (sección 8.5, decisión O7);
- vida media dentro de un rango útil (avisos si supera 126 días o si es menor que 1);
- residuos sin autocorrelación (Ljung–Box, sección 8.7);
- estabilidad de `λ` entre ventanas (sección 8.7: se estima en cada ventana con
  `0 < b < 1`; la prueba de raíz unitaria por ventana es un dato aparte);
- costes pequeños frente a `σ_eq` (sección 15).

**En el código:** `mrlab.stats.correlation` (correlaciones),
`mrlab.data.providers.synthetic` (escenarios pedagógicos).

---

## 3. Series I(0) e I(1): pruebas ADF y KPSS

### 3.1 Intuición

- **I(0), estacionaria:** fluctúa alrededor de un nivel fijo con una dispersión
  estable. Los choques se desvanecen con el tiempo. Ejemplos: los rendimientos
  diarios; el spread de un par cointegrado.
- **I(1), integrada de orden 1:** no tiene un nivel al que volver. Los choques son
  permanentes y la dispersión crece con el tiempo, pero sus **primeras
  diferencias** sí son I(0). Ejemplo típico: los log-precios, que se parecen a un
  paseo aleatorio.
- **Cointegración:** dos series I(1) tales que una combinación lineal de ellas es
  I(0). Por eso primero se verifica que **cada serie** sea I(1): Engle–Granger
  supone series I(1), y si una ya fuera estacionaria, el resultado de la prueba
  sería engañoso.

La conexión con el modelo AR(1) aclara todo lo que sigue:

```text
x_t = c + b·x_{t−1} + e_t
b = 1   → paseo aleatorio (raíz unitaria, I(1))
|b| < 1 → estacionaria (I(0))
Restando x_{t−1} de ambos lados:  Δx_t = c + γ·x_{t−1} + e_t,   con γ = b − 1
```

Preguntar «¿hay raíz unitaria?» equivale a preguntar «¿`γ = 0`?».

### 3.2 Prueba ADF (Dickey–Fuller aumentada)

```text
Δx_t = c + γ·x_{t−1} + Σ_{j=1..p} φ_j·Δx_{t−j} + e_t

H0: γ = 0  (raíz unitaria: NO estacionaria)
H1: γ < 0  (estacionaria)
Estadístico: τ = γ̂ / EE(γ̂)
```

- Los términos `Δx_{t−j}` absorben la autocorrelación de corto plazo. El número de
  rezagos `p` se elige por AIC, con un máximo de 12. La regresión lleva constante.
- Bajo H0, `τ` **no** sigue una t de Student. Sus valores críticos son los de
  Dickey–Fuller, calculados por MacKinnon: con constante, el valor crítico al 5 %
  es ≈ −2,86 (una t unilateral sería −1,65).
- Se rechaza H0 si `p ≤ α` (por defecto `α = 0,05`).

### 3.3 Prueba KPSS

```text
x_t = c + r_t + u_t,    r_t = r_{t−1} + v_t   (paseo aleatorio con varianza σ_v²)

H0: σ_v² = 0  (estacionaria alrededor de c)
H1: σ_v² > 0  (tiene raíz unitaria)
Estadístico: η = (1/n²)·Σ_t S_t² / σ̂²_LP,   con S_t = Σ_{s≤t} (x_s − x̄)
```

- **Intuición:** si la serie es estacionaria, la suma acumulada de sus desvíos
  respecto de la media se mantiene pequeña. Si deambula, esas sumas crecen mucho.
- `σ̂²_LP` es la varianza de largo plazo (Newey–West, con rezagos automáticos de
  Hobijn et al.). La regresión lleva constante.
- La tabla de KPSS solo permite interpolar el p-value entre 0,01 y 0,10 (decisión
  E1). Fuera de ese rango se informa «p < 0,01» (rechaza si `α ≥ 0,01`) o
  «p > 0,10» (nunca rechaza).

### 3.4 Hipótesis nulas opuestas e interpretación combinada (decisión E2)

**«No rechazar» no es lo mismo que «aceptar».** Con pocos datos, ADF suele no
rechazar la raíz unitaria aunque la serie sea estacionaria: tiene poca potencia.
KPSS parte de la hipótesis contraria. Si las dos pruebas apuntan a la misma
conclusión, la conclusión es más sólida que la de una sola.

Se aplican a cada serie **en niveles** (log-precios, o precios si el spread es en
precios) y **en primeras diferencias**, solo con la muestra de formación.

| Niveles: ADF | Niveles: KPSS | Diferencias: ADF | Diferencias: KPSS | Clasificación |
|---|---|---|---|---|
| No rechaza | Rechaza | Rechaza | No rechaza | **Compatible con I(1)** |
| Rechaza | No rechaza | (cualquiera) | (cualquiera) | **Posiblemente estacionaria** |
| Cualquier otra combinación | | | | **Resultado ambiguo** |

- **Compatible con I(1):** en niveles las dos pruebas dicen «no estacionaria» y en
  diferencias las dos dicen «estacionaria». Es lo que se espera de un precio.
- **Ambiguo:** por ejemplo, las dos rechazan en niveles (es habitual con cambios
  estructurales o muestras muy largas), o ninguna rechaza (pocos datos).
- Si alguna de las dos series **no** es «compatible con I(1)», el par queda en
  **⚠️ Supuestos dudosos**.

Ejemplo ilustrativo (valores inventados para una acción típica): en niveles,
ADF p = 0,41 (no rechaza) y KPSS p < 0,01 (rechaza); en diferencias, ADF
p < 0,0001 (rechaza) y KPSS p > 0,10 (no rechaza). Clasificación: compatible
con I(1).

**En el código:** `mrlab.stats.integration` (`adf_test`, `kpss_test`,
`integration_tests`).

---

## 4. Cointegración: prueba de Engle–Granger

### 4.1 Los dos pasos

```text
Paso 1 (MCO):   y_t = α + β·x_t + u_t   →   residuos û_t = y_t − α̂ − β̂·x_t

Paso 2 (ADF sobre los residuos, sin constante):
                Δû_t = γ·û_{t−1} + Σ_{j=1..p} φ_j·Δû_{t−j} + e_t

H0: γ = 0  (NO hay cointegración: el residuo tiene raíz unitaria)
H1: γ < 0  (cointegración: el residuo es estacionario)
```

El residuo `û_t` es, precisamente, el spread candidato. La regresión del paso 1
lleva constante (`trend = "c"`); el ADF del paso 2 no la lleva porque los
residuos de MCO ya tienen media cero. Los rezagos se eligen por AIC, con máximo 12.

### 4.2 Por qué p-values de MacKinnon para residuos estimados

**Intuición.** MCO elige `α` y `β` para que los residuos sean lo más pequeños
posible. Con eso, los residuos **parecen** más estacionarios de lo que son,
incluso entre dos paseos aleatorios independientes. Aplicarles un ADF ordinario es
jugar con un mazo marcado: la prueba rechaza H0 con demasiada frecuencia y
«descubre» cointegración donde no la hay. Las tablas de MacKinnon para residuos
estimados (con N = 2 variables) corrigen ese sesgo con valores críticos más
negativos.

| Valores críticos asintóticos (con constante) | 1 % | 5 % | 10 % |
|---|---|---|---|
| ADF ordinario | −3,43 | −2,86 | −2,57 |
| Engle–Granger, 2 variables | −3,90 | −3,34 | −3,04 |

(El laboratorio muestra los valores críticos para el `n` real de tu muestra.)

**Ejemplo.** El estadístico sobre los residuos vale `τ = −3,00`.

| Tabla usada | p-value | Conclusión al 5 % |
|---|---|---|
| ADF ordinario (incorrecto) | ≈ 0,035 | «Hay cointegración» |
| Engle–Granger / MacKinnon (correcto) | ≈ 0,110 | Evidencia insuficiente |

El mismo número lleva a conclusiones opuestas. Por eso el laboratorio toma el
p-value **exclusivamente** de `statsmodels.tsa.stattools.coint`, y una prueba
automática verifica que no coincide con el de un ADF ordinario sobre los residuos.

**Caso especial.** Si las dos series son prácticamente idénticas salvo escala
(R² ≈ 1), `coint` devuelve un estadístico −∞ y p = 0. Ese p = 0 **no** es evidencia
fiable: el spread es solo ruido de redondeo. El laboratorio no ajusta el OU y
descarta el par con `SERIES_COLINEALES`, que bloquea la simulación (decisión O8).
Si es literalmente el mismo activo (mismo proveedor y símbolo), el estado es ⛔
con la causa `MISMO_ACTIVO` (decisión D8). En una ventana de estabilidad, un
estadístico −∞ cuenta como ventana no evaluable (sección 6.2).

### 4.3 Asimetría: las dos direcciones

Regresar A sobre B no es lo mismo que regresar B sobre A. Las pendientes no son
recíprocas:

```text
β̂_{A sobre B} · β̂_{B sobre A} = R²     (solo si R² = 1 se cumple β̂_{B sobre A} = 1/β̂_{A sobre B})
```

Los residuos son distintos, y por tanto también el estadístico y el p-value. El
laboratorio calcula **ambas direcciones** (decisión E3):

- por defecto usa la de **menor p-value** y lo indica en la interfaz;
- el usuario puede fijar la dirección;
- la dirección elegida define cuál es **A** (la variable dependiente) en el spread.

Elegir el menor de dos p-values es en sí una pequeña selección: si se hacen dos
intentos, es más fácil que uno salga «significativo» por azar. Por eso, incluso
con un solo par, se aplica Benjamini–Hochberg sobre las dos direcciones (sección
5.4).

**En el código:** `mrlab.stats.cointegration` (`engle_granger`,
`engle_granger_both`, `choose_direction`, `eg_decision`).

---

## 5. Pruebas múltiples: Benjamini–Hochberg

### 5.1 El problema

Con un umbral del 5 %, cada prueba sobre un par **no** cointegrado tiene un 5 % de
probabilidad de dar un falso positivo. Con 100 pares independientes sin relación
alguna esperarías unos 5 «descubrimientos» falsos. Con 30 activos hay 435 pares y
870 direcciones: unos 43 falsos positivos esperados aunque ningún par estuviera
cointegrado.

Benjamini–Hochberg (BH) controla la **tasa de falsos descubrimientos** (FDR): la
proporción esperada de falsos positivos entre los pares que declaras
significativos.

```text
FDR = E[ V / max(R, 1) ]     V = falsos descubrimientos,  R = descubrimientos totales
```

Bonferroni controla algo más exigente (la probabilidad de **algún** falso
positivo) y pierde mucha potencia. BH es un punto intermedio razonable para
explorar un universo.

### 5.2 Procedimiento y p-value ajustado

```text
1. Ordenar los m p-values:      p_(1) ≤ p_(2) ≤ … ≤ p_(m)
2. Buscar el mayor k tal que:   p_(k) ≤ q·k/m
3. Rechazar H_(1), …, H_(k)

p-value ajustado:   p̃_(i) = min_{k ≥ i} min(1, m·p_(k)/k)
Regla equivalente:  se rechaza H_(i) si p̃_(i) ≤ q
```

- El `min` sobre `k ≥ i` garantiza que los p ajustados conserven el orden de los
  originales.
- Los p ajustados **no dependen de `q`**: `q` solo decide qué se rechaza.
- En el laboratorio hay **evidencia** si el p ajustado de la dirección elegida es
  ≤ el nivel de significancia (0,05 por defecto). Siempre se muestran el p
  original y el ajustado lado a lado.

### 5.3 Ejemplo con m = 10 (5 pares × 2 direcciones), q = 0,05

| k | p_(k) | Umbral q·k/m | m·p_(k)/k | p ajustado | ¿Rechaza? |
|---|---|---|---|---|---|
| 1 | 0,001 | 0,005 | 0,0100 | 0,0100 | Sí |
| 2 | 0,004 | 0,010 | 0,0200 | 0,0200 | Sí |
| 3 | 0,012 | 0,015 | 0,0400 | 0,0400 | Sí |
| 4 | 0,030 | 0,020 | 0,0750 | 0,0750 | No |
| 5 | 0,045 | 0,025 | 0,0900 | 0,0900 | No |
| 6 | 0,090 | 0,030 | 0,1500 | 0,1500 | No |
| 7 | 0,200 | 0,035 | 0,2857 | 0,2857 | No |
| 8 | 0,350 | 0,040 | 0,4375 | 0,4375 | No |
| 9 | 0,600 | 0,045 | 0,6667 | 0,6667 | No |
| 10 | 0,850 | 0,050 | 0,8500 | 0,8500 | No |

| Criterio | Descubrimientos |
|---|---|
| Sin corrección (p ≤ 0,05) | 5 |
| Benjamini–Hochberg (p ajustado ≤ 0,05) | 3 |
| Bonferroni (p ≤ 0,05/10 = 0,005) | 2 |

### 5.4 Un solo par: m = 2 (decisión E4)

Al analizar un solo par, la familia son sus dos direcciones.

| p (A sobre B) | p (B sobre A) | Cálculo | p ajustados | Decisión con la dirección de menor p |
|---|---|---|---|---|
| 0,030 | 0,400 | 2·0,030/1 = 0,060 ; 2·0,400/2 = 0,400 | 0,060 ; 0,400 | Evidencia insuficiente (0,060 > 0,05) |
| 0,030 | 0,045 | 0,060 ; 0,045 → el mínimo hacia la derecha deja 0,045 | 0,045 ; 0,045 | Evidencia (0,045 ≤ 0,05) |
| 0,020 | 0,600 | 0,040 ; 0,600 | 0,040 ; 0,600 | Evidencia (0,040 ≤ 0,05) |

El primer caso es pedagógico: un p original de 0,030 «parece significativo», pero
después de reconocer que se hicieron dos intentos ya no supera el umbral.

En un universo, la familia son **todos los pares × 2 direcciones** evaluados.

### 5.5 Benjamini–Yekutieli (opción)

BH supone pruebas independientes o con dependencia positiva (las dos direcciones
de un mismo par lo están). Benjamini–Yekutieli es válido con **cualquier**
dependencia, a cambio de ser más conservador:

```text
c(m) = Σ_{k=1..m} 1/k            (c(2) = 1,5 ;  c(870) ≈ 7,35)
p̃_(i) = min_{k ≥ i} min(1, c(m)·m·p_(k)/k)
```

**En el código:** `mrlab.stats.multitest` (`benjamini_hochberg`,
`benjamini_yekutieli`, `adjust_pvalues`). Una prueba automática verifica que el
resultado coincide con `statsmodels.stats.multitest.multipletests`.

---

## 6. Estabilidad entre ventanas y estado del par

### 6.1 Intuición

Una relación que aparece en toda la muestra de formación puede deberse a un solo
periodo. Para comprobarlo se repite Engle–Granger en ventanas de 252
observaciones (≈ 1 año) que avanzan de 63 en 63 (≈ 1 trimestre), siempre con la
**dirección fija** elegida antes.

### 6.2 Reglas

```text
Número de ventanas = floor((n − 252)/63) + 1      (si n ≥ 252; si no, no se evalúa)
Una ventana FALLA si su p-value > α                (p SIN ajustar: es un diagnóstico, decisión E5)
Resultado INESTABLE si:  fracción de ventanas fallidas > 40 %   o   β cambia de signo
```

**Ejemplo.** Con 1.500 observaciones de formación hay
`floor(1.248/63) + 1 = 20` ventanas. El resultado es inestable si fallan 9 o más
(9/20 = 45 % > 40 %), o si en alguna ventana `β` es positivo y en otra negativo
(la relación se invirtió). Con el mínimo de 500 observaciones hay solo 4 ventanas.

**Ventanas no evaluables (decisión E10).** Si en una ventana Engle–Granger no se
puede calcular (por ejemplo, una serie que no cambia de precio durante toda la
ventana) o da un estadístico −∞ (series colineales en esa ventana), la ventana
cuenta como **fallida**, con su motivo y sus fechas. No se convierte en un error
de todo el par: es un diagnóstico. El cambio de signo de `β` se evalúa solo con
las ventanas que tienen `β`.

**Poca potencia.** Con 252 observaciones, Engle–Granger detecta mal incluso
relaciones reales. Es esperable que muchos pares reales salgan «inestables»: eso
es información sobre lo frágil de la evidencia, no un error del laboratorio.

### 6.3 Del resultado de las pruebas al estado del par

Hay exactamente cinco estados. Si se activan varias causas, el estado principal
sigue esta precedencia (decisión E6), y **siempre** se listan todas las causas:

```text
⛔ Datos insuficientes  >  ⚠️ Supuestos dudosos  >  ⚠️ Evidencia insuficiente
                        >  ⚠️ Resultado inestable entre ventanas  >  ✅ Evidencia de cointegración en la ventana evaluada
```

✅ significa «hay evidencia en la ventana evaluada», nunca «cointegración
garantizada». El estado ⛔ cubre tanto la falta de datos (menos de 500
observaciones en formación) como los bloqueos por datos no aptos (monedas
distintas o desconocidas, precios ≤ 0 con spread logarítmico, el mismo activo en
las dos piernas).

El estado describe la **evidencia estadística**. Aparte, el panel «¿Por qué se
descartó este par?» lista las razones de aptitud para operar (OU no válido, vida
media fuera de [1, 126], series colineales, proxy ejecutable requerido, las dos
piernas operadas con el mismo instrumento, precios ejecutables ≤ 0, costes que
superan el PnL bruto…). Cualquier razón de ese panel,
o un estado distinto de ✅, hace que el backtest sea **solo ilustrativo**, con un
aviso que cita los motivos (decisión O5a).

**En el código:** `mrlab.stats.cointegration.rolling_eg_stability`,
`mrlab.stats.windows.rolling_windows`, `mrlab.analysis.pair_pipeline`.

---

## 7. El spread y el hedge ratio

### 7.1 Dos definiciones del spread

```text
Logarítmico (por defecto):  X_t = ln P_A,t − α − β·ln P_B,t      [log-puntos]
En precios:                 X_t = P_A,t − α − β·P_B,t            [moneda]
```

- `A` es la variable dependiente de la dirección elegida en Engle–Granger.
- `α` y `β` se estiman por MCO **solo con la muestra de formación** y quedan
  **fijos** en validación y prueba.
- **¿Por qué logarítmico por defecto?** Es invariante a la escala: un spread de
  1 USD significa cosas muy distintas si la acción vale 20 o 200, mientras que
  un 1 % es un 1 % en ambos casos. Además, `β` se interpreta como elasticidad.
  Exige precios estrictamente positivos (si no, el análisis se bloquea).
- **Spread en precios:** `β` se interpreta en unidades. Su volatilidad crece con
  el nivel de los precios, por lo que es más útil cuando ambos activos tienen
  precios parecidos y estables.

Estimación por MCO (con `y = f(P_A)`, `x = f(P_B)`, donde `f` es `ln` o la
identidad):

```text
β̂ = Cov(y, x) / Var(x)
α̂ = ȳ − β̂·x̄
Primera ecuación normal de MCO:  Σ_t û_t = 0   →   el spread tiene media EXACTAMENTE 0 en formación
```

### 7.2 El hedge ratio β: elasticidad frente a unidades

| Spread | Significado de β | Qué relaciona |
|---|---|---|
| Logarítmico | Elasticidad: un 1 % en B se asocia con un β % en A | **Nocionales:** por cada 1 USD en A, β USD en B |
| En precios | β unidades de B por cada unidad de A | **Cantidades:** por cada acción de A, β acciones de B |

**Error frecuente:** usar el `β` del spread logarítmico como si fueran unidades.
Con A a 50 USD, B a 200 USD y `β_log = 1`:

| | Pierna A | Pierna B | Resultado |
|---|---|---|---|
| Correcto (nocionales iguales) | 100 acciones = 5.000 USD | 25 acciones = 5.000 USD | Cobertura equilibrada |
| Incorrecto («β unidades») | 100 acciones = 5.000 USD | 100 acciones = 20.000 USD | La posición la domina B |

### 7.3 Consistencia α–μ: no restar dos veces el mismo nivel

Como `α` ya está dentro del spread, el residuo de formación tiene media cero. El
`μ` del OU se estima sobre **ese** residuo y se usa **tal cual** en el z-score.

`μ̂` sale ≈ 0, pero no exactamente 0, porque el AR(1) usa pares consecutivos
`(X_t, X_{t+1})`. Con `n` pares, `x̄₀` la media de `X_0..X_{n−1}` y `x̄₁` la de
`X_1..X_n`:

```text
μ̂ = â/(1 − b̂) = (x̄₁ − b̂·x̄₀)/(1 − b̂) = x̄₀ + (X_n − X_0) / (n·(1 − b̂))
```

Ambos términos son pequeños: `x̄₀ ≈ 0` porque el residuo tiene media cero, y el
segundo término decrece con `n`.

**El error que se evita:** restar el nivel dos veces. Por ejemplo, calcular
`z = (X_t − α̂)/σ_eq` con un `X_t` que ya tenía `α̂` restado, o estimar `μ` sobre
`ln A − β·ln B` (cuya media es ≈ `α̂`) y restarlo de un spread que ya excluía
`α̂`. Con `α̂ = 0,30` y `σ_eq = 0,05`:

```text
z_erróneo = z_correcto − 0,30/0,05 = z_correcto − 6
```

Cuando el spread está justo en su referencia (z correcto = 0), la pantalla
mostraría z = −6: una señal permanente de «comprar el spread» que además dispara el
stop de inmediato. Hay una prueba automática que verifica que `μ̂` estimado sobre
el residuo de formación es ≈ 0.

Una alternativa equivalente y válida: definir `X' = ln A − β·ln B` sin `α`. En ese
caso `μ̂' ≈ α̂` y el z-score es el mismo. El nivel va en el spread **o** en `μ`,
nunca en los dos.

### 7.4 Aviso sobre la inferencia MCO con series cointegradas (decisión E7)

- **Regresión espuria** (Granger y Newbold, 1974): si regresas un paseo aleatorio
  sobre otro **independiente**, es frecuente obtener un R² alto y un `β`
  «significativo» (|t| > 2). Por eso un R² alto **no prueba nada**.
- **Superconsistencia:** si las series sí están cointegradas, `β̂` converge al
  valor verdadero más rápido de lo habitual (su error se reduce como `1/n` en
  lugar de `1/√n`), pero su distribución no es normal y los errores estándar
  usuales de MCO no la describen bien (los residuos están autocorrelacionados y
  el regresor no es exógeno).

El laboratorio muestra el error estándar de `β` y el R² solo como referencia
descriptiva, con este aviso. La evidencia de cointegración la da Engle–Granger.
Estimadores como DOLS o FM-OLS quedan como extensión.

**En el código:** `mrlab.model.spread` (`transform`, `fit_hedge_ratio`,
`compute_spread`).

---

## 8. El proceso de Ornstein–Uhlenbeck

### 8.1 La ecuación y la intuición de cada parámetro

```text
dX_t = λ·(μ − X_t)·dt + σ·dW_t        (λ ≡ θ en otras notaciones)
```

**Intuición: una banda elástica.** Cuanto más se aleja el spread de `μ`, más
fuerte es el tirón que lo devuelve (el término `λ·(μ − X_t)`). Al mismo tiempo, un
ruido aleatorio (`σ·dW_t`) lo empuja en cualquier dirección. El spread vive en el
equilibrio entre esas dos fuerzas.

| Parámetro | Qué representa | Unidades |
|---|---|---|
| `λ` | Velocidad de reversión. Cada día, la desviación **esperada** respecto de `μ` se multiplica por `e^{−λ}` (para `λ` pequeño, se reduce ≈ `λ·100 %`) | 1/día de negociación |
| `μ` | **Referencia estadística del modelo:** el nivel alrededor del cual fluctuó el spread en formación. **No** es un «valor justo» ni un objetivo de precio | unidades del spread |
| `σ` | Intensidad del ruido | unidades del spread / √día |
| `σ_eq = σ/√(2λ)` | Tamaño típico de las desviaciones a largo plazo | unidades del spread |
| Vida media `= ln 2/λ` | Tiempo en que la desviación **esperada** se reduce a la mitad | días de negociación |

**De dónde sale `σ_eq`.** El ruido añade varianza a ritmo `σ²` por unidad de
tiempo, y el tirón la elimina a ritmo `2λ·Var`. En equilibrio, ambas se igualan:

```text
dVar/dt = σ² − 2λ·Var = 0    →    Var = σ²/(2λ)    →    σ_eq = σ/√(2λ)
```

Más ruido ensancha el rango del spread; una reversión más rápida lo estrecha.

### 8.2 La transición exacta (derivación)

Se multiplica `X_t` por el factor integrante `e^{λt}`. Por la regla de Itô (aquí no
aparece término extra porque `e^{λt}` es determinista y lineal en `X`):

```text
d(e^{λt}·X_t) = e^{λt}·(λ·X_t·dt + dX_t) = e^{λt}·(λ·μ·dt + σ·dW_t)
```

Integrando entre `t` y `t + Δt` y dividiendo por `e^{λ(t+Δt)}`:

```text
X_{t+Δt} = μ + (X_t − μ)·e^{−λΔt} + σ·∫_t^{t+Δt} e^{−λ(t+Δt−s)} dW_s
```

La integral estocástica es normal con media 0 y, por la isometría de Itô,
varianza `σ²·∫_0^{Δt} e^{−2λu} du = σ²·(1 − e^{−2λΔt})/(2λ)`. Entonces:

```text
X_{t+Δt} = μ + (X_t − μ)·e^{−λΔt} + ε_t,     ε_t ~ N(0, σ_eq²·(1 − e^{−2λΔt}))
```

Esto es exactamente un **AR(1)**:

```text
X_{t+Δt} = a + b·X_t + ε_t,     ε_t ~ N(0, s²)
b  = e^{−λΔt}
a  = μ·(1 − b)
s² = σ_eq²·(1 − b²) = σ²·(1 − b²)/(2λ)
```

**Exacta, no aproximada.** No hay error de discretización para ningún `Δt`. El
laboratorio la usa tanto para estimar como para simular. La aproximación de Euler
(`b ≈ 1 − λΔt`) se degrada cuando la reversión es rápida:

| `λΔt` | Euler `1 − λΔt` | Exacta `e^{−λΔt}` |
|---|---|---|
| 0,05 | 0,9500 | 0,9512 |
| 0,50 | 0,5000 | 0,6065 |

### 8.3 El estimador AR(1)

Se estima el AR(1) por MCO con constante sobre los pares consecutivos
`(X_t, X_{t+1})` del spread de formación, y luego se invierten las relaciones de
la sección 8.2 (decisión O1):

```text
b̂  = Σ (X_t − x̄₀)·(X_{t+1} − x̄₁) / Σ (X_t − x̄₀)²
â  = x̄₁ − b̂·x̄₀
s² = Σ ε̂_t² / (n − 2)                 (n = número de pares; ε̂_t = X_{t+1} − â − b̂·X_t)

Si 0 < b̂ < 1:
  λ̂     = −ln(b̂) / Δt
  μ̂     = â / (1 − b̂)
  σ̂     = sqrt( 2·λ̂·s² / (1 − b̂²) )
  σ̂_eq  = σ̂ / sqrt(2·λ̂) = s / sqrt(1 − b̂²)
  vida media = ln 2 / λ̂
```

Cada fórmula sale de despejar la sección 8.2: `b = e^{−λΔt}` da `λ`;
`a = μ(1 − b)` da `μ`; `s² = σ²(1 − b²)/(2λ)` da `σ`. Con `Δt` constante, MCO
sobre el AR(1) coincide con la máxima verosimilitud condicional. Solo se usan
pares con ambos valores disponibles (un hueco nunca crea una transición
ficticia) y se exigen al menos 30 observaciones.

**Ejemplo.** Un spread logarítmico da `â = 0,0004`, `b̂ = 0,95`, `s = 0,0195`:

| Parámetro | Cálculo | Resultado |
|---|---|---|
| `λ̂` | `−ln 0,95` | 0,0513 por día |
| `μ̂` | `0,0004 / 0,05` | 0,008 log-puntos |
| `σ̂` | `sqrt(2·0,0513·0,0195² / (1 − 0,9025))` | 0,0200 log-puntos/√día |
| `σ̂_eq` | `0,0195 / sqrt(0,0975)` | 0,0624 log-puntos (≈ 6,2 %) |
| Vida media | `0,693 / 0,0513` | 13,5 días de negociación |

### 8.4 Media y varianza condicionales

Con `Δt = h` en la sección 8.2:

```text
E[X_{t+h} | X_t]   = μ + (X_t − μ)·e^{−λh}
Var[X_{t+h} | X_t] = σ_eq²·(1 − e^{−2λh})

Cuando h → ∞:  E → μ,  Var → σ_eq²    (distribución estacionaria N(μ, σ_eq²))
```

**Vida media.** Es el `h` en que la desviación esperada cae a la mitad:
`(X_t − μ)·e^{−λh} = ½·(X_t − μ)`, de donde `h = ln 2/λ`.

**Ejemplo.** `λ = 0,05` por día, `σ = 0,02`, `μ = 0`, entonces
`σ_eq = 0,02/√0,1 = 0,0632`, y vida media = 13,9 días. Partimos de
`X_t = 2·σ_eq = 0,1265` (z = 2):

| h (días) | Media condicional | z esperado | Desviación condicional | Intervalo 95 % |
|---|---|---|---|---|
| 1 | 0,1203 | 1,90 | 0,0195 | [0,0821 ; 0,1586] |
| 5 | 0,0985 | 1,56 | 0,0397 | [0,0208 ; 0,1763] |
| 14 | 0,0628 | 0,99 | 0,0549 | [−0,0448 ; 0,1704] |
| 20 | 0,0465 | 0,74 | 0,0588 | [−0,0687 ; 0,1618] |
| 60 | 0,0063 | 0,10 | 0,0632 | [−0,1175 ; 0,1301] |

Lectura: la reversión es una **tendencia**, no una promesa. A 20 días, la media
esperada sigue siendo positiva, pero el intervalo ya incluye valores negativos
(el spread pudo cruzar `μ`) y valores más altos que el de partida.

**Tiempo esperado de una operación.** El camino medio va de `z_e` a `z_s` en
`h = ln(z_e/z_s)/λ`. De 2,0 a 0,5: `ln 4/λ` = 2 vidas medias (27,7 días con
`λ = 0,05`). El tiempo máximo por defecto, 3 vidas medias (sección 11), deja
margen sobre ese valor medio.

### 8.5 Cuándo el ajuste OU es válido (decisión O7)

En un OU, `b = e^{−λΔt}` con `λ > 0`, así que **siempre** `0 < b < 1`.

- **`b̂ ≥ 1`:** implicaría `λ = −ln b ≤ 0`, es decir, ningún tirón hacia `μ`
  (`b = 1`, paseo aleatorio) o un empuje que aleja (`b > 1`). La vida media sería
  infinita o negativa y `1 − b² ≤ 0` hace imposible calcular `σ`. Estado: «no
  estacionario».
- **`b̂ ≤ 0`:** `ln b` no existe (`b < 0`) o es −∞ (`b = 0`). Un `b` negativo
  significa que el spread tiende a cambiar de signo de una observación a la
  siguiente (oscila), algo que un OU observado a intervalos regulares no puede
  producir.

En ambos casos el laboratorio muestra «el spread no muestra reversión
estacionaria compatible con un OU» y **no** calcula `λ`, `μ`, `σ`, vida media,
bandas ni z-score. Sin z no hay estrategia (decisión O5).

**`0 < b̂ < 1` no basta: el sesgo de Dickey–Fuller.** Cuando el verdadero `b` es
1 (paseo aleatorio), MCO lo subestima: `b̂` queda por debajo de 1 casi siempre.
Si la validez fuera solo `0 < b̂ < 1`, un paseo aleatorio saldría como «OU
válido» con una vida media finita y engañosa (por ejemplo, `b̂ = 0,99` da una vida
media de 69 días). Por eso el laboratorio exige además que la prueba de
Dickey–Fuller **aumentada** (ADF, sección 3.2) **rechace** la raíz unitaria:

```text
Δx_t = [c] + γ·x_{t−1} + Σ_{j=1..p} φ_j·Δx_{t−j} + e_t     (p elegido por AIC, máximo 12)
τ = γ̂ / EE(γ̂)

Válido  ⇔  0 < b̂ < 1   y   p-value(τ) ≤ α            (equivale a τ ≤ c_α)
```

`α` es el nivel de evidencia del análisis (`alpha_level`, 0,05 por defecto). La
constante `[c]` solo se incluye para una serie observada (ver la tabla de valores
críticos más abajo). El reparto de tareas es claro: el AR(1) de la sección 8.3 **estima** `a`, `b`, `λ`,
`μ` y `σ`; el ADF **decide** si hay evidencia de que el spread revierta.

**¿Por qué con rezagos?** Si los cambios del spread tienen memoria, la prueba sin
rezagos (`τ = (b̂ − 1)/EE(b̂)` del AR(1)) se equivoca. El caso típico es el ruido
de microestructura: un cierre con un pequeño error de medición. Si
`x_t = paseo_t + ruido_t`, los cambios `Δx_t` tienen autocorrelación negativa (con
la misma volatilidad en el paseo y en el ruido, `ρ₁ = −1/3`): el ruido de hoy se
deshace mañana, y eso parece reversión. Los términos `Δx_{t−j}` del ADF absorben
esa memoria de corto plazo.

**Ejemplo** (paseo aleatorio con volatilidad diaria del 1 % más un ruido de
medición del 1 %, 900 observaciones, semilla 11; lo reproduce el código del
laboratorio):

| Prueba | Estadístico | Rezagos | p-value | Conclusión al 5 % |
|---|---|---|---|---|
| Dickey–Fuller sin rezagos (regla anterior) | −3,93 | 0 | 0,0018 | «OU válido», vida media 20,0 días |
| ADF con rezagos por AIC (regla actual) | −2,31 | 2 | 0,170 | No se distingue de un paseo aleatorio |

La autocorrelación de los cambios es −0,34: el AR(1) ve un `b̂ = 0,966` que no
significa nada, porque la serie no tiene ningún nivel al que volver.

**¿Qué valores críticos?** Dependen de cómo se obtuvo la serie, por la misma
razón de la sección 4.2:

| Serie a la que se aplica la prueba | Regresión ADF | p-values de MacKinnon | `c` al 5 % |
|---|---|---|---|
| Serie observada (p. ej. una simulación) | con constante | Dickey–Fuller, N = 1 | −2,86 |
| Spread del par: **residuo** de la regresión de formación (α y β estimados con los mismos datos) | sin constante (el residuo de MCO ya tiene media cero) | Engle–Granger, N = 2 | −3,34 |

MCO elige `α` y `β` para que el residuo sea lo más pequeño posible, así que el
spread de un par parece más estacionario de lo que es: con la tabla N = 1 la
prueba lo aceptaría demasiadas veces. Para el spread del par, la prueba del
laboratorio es **exactamente el paso interno de Engle–Granger** (`coint` aplica
ese mismo ADF sin constante al residuo y usa esa misma tabla). Por eso:

```text
OU válido sobre el spread de formación  ⇔  0 < b̂ < 1  y  p de Engle–Granger SIN AJUSTAR (dirección elegida) ≤ α
```

Se decide con el p-value. El valor crítico que se muestra es el que corresponde
exactamente a ese p-value (se obtiene invirtiendo la fórmula de MacKinnon), así
que «estadístico ≤ valor crítico» y «p-value ≤ α» nunca se contradicen. La tabla
con corrección de muestra finita da casi lo mismo (con 899 transiciones, −3,343
frente a −3,336 al 5 % con N = 2), pero haría discrepar al OU de Engle–Granger
justo en esa franja.

**Ejemplo** (la demo con el par sintético SINT-COI): Engle–Granger de A sobre B da
un estadístico −6,611 con 0 rezagos; el ADF del spread da −6,61 con 0 rezagos,
frente a un valor crítico de −3,34 al 5 % y un p-value < 0,0001. Es el mismo
número porque es el mismo cálculo.

**Relación con el estado del par.** Engle–Granger decide el estado con el p-value
**ajustado** por Benjamini–Hochberg (sección 5.4), que nunca es menor que el p sin
ajustar. Por tanto, un par ✅ siempre supera la prueba de raíz unitaria del OU.
Lo contrario no es cierto: si `p sin ajustar ≤ α < p ajustado`, el OU es válido
pero el estado es ⚠️ Evidencia insuficiente, y el backtest se muestra solo como
ilustración (decisión O5).

**Comprobación por simulación** (2.000 repeticiones de 900 observaciones, como una
formación de 1.500 con la división 60/20/20; porcentaje de casos que salen «OU
válido»; lo ideal es ≈ 5 %):

| Caso | `0 < b̂ < 1` (regla original) | Tabla | Sin rezagos (regla anterior) | ADF (regla actual) |
|---|---|---|---|---|
| Paseo aleatorio observado | 96,2 % | N = 1 | 4,8 % | 5,3 % |
| Residuo de MCO entre dos paseos independientes | 99,1 % | N = 1 (incorrecta) | 13,3 % | 14,3 % |
| Residuo de MCO entre dos paseos independientes | 99,1 % | N = 2 | 4,8 % | 5,4 % |
| Paseo aleatorio + ruido de medición del 0,5 % | 98,8 % | N = 1 | 11,9 % | 5,2 % |
| Paseo aleatorio + ruido de medición del 1 % | 100,0 % | N = 1 | 38,0 % | 5,7 % |

Con la regla original, casi todos los paseos aleatorios pasaban como OU válidos.
Con la tabla correcta y los rezagos, la tasa de falsos «OU válidos» queda cerca
del 5 % nominal, también con ruido de microestructura. En el análisis completo de
200 pares de paseos aleatorios independientes de 1.500 observaciones, con la regla
original los 200 daban un OU «válido»; con la regla actual, 21 (10,5 %) y ninguno
✅. Supera el 5 % porque el laboratorio elige la dirección de Engle–Granger con
menor p-value, y eso favorece el spread que más parece estacionario. En los 200
casos la validez del OU coincidió con Engle–Granger sin ajustar.

**Si la prueba no rechaza** (`near_unit_root`: `0 < b̂ < 1` pero `b̂` no se
distingue de 1): el estado del ajuste es «no estacionario», se muestran `b̂`, el
estadístico, sus rezagos, su p-value y el valor crítico, y no se calculan `λ`,
`μ`, `σ`, vida media ni bandas. El par recibe la razón de descarte
`OU_NO_ESTACIONARIO`, que bloquea la estrategia. Si la prueba no se puede calcular
(por ejemplo, con faltantes se usa el tramo continuo más largo, y tiene menos de
30 observaciones), el ajuste es «datos insuficientes».

**Ruido de redondeo.** Si las dos piernas son idénticas o proporcionales, el
spread es casi nulo y solo contiene ruido numérico. El laboratorio compara la
desviación del spread con la escala de las series (la desviación de `ln A` en
formación): si es menor que `10⁻¹⁰` veces esa escala, el ajuste es «datos
insuficientes» (decisión O8).

**Avisos de vida media:** más de 126 días (medio año de negociación) es
demasiado lenta para operar con este horizonte. Menos de 1 día no es operable ni
fiable con datos diarios. Por ejemplo, `b̂ = 0,4` da `λ̂ = 0,92` y una vida media
de 0,76 días.

### 8.6 Error estándar de μ̂ (decisiones O1 y O2)

**Intuición.** En un spread que revierte despacio, dos observaciones seguidas son
casi la misma información. Para estimar el nivel medio, 5.000 días de un spread
lento valen mucho menos que 5.000 datos independientes.

Las autocorrelaciones de un AR(1) estacionario son `b^k`, y la varianza de la
media muestral es:

```text
Var(x̄) ≈ (σ_eq²/n)·(1 + 2·Σ_{k≥1} b^k) = (σ_eq²/n)·(1 + b)/(1 − b)

EE(μ̂)       = σ_eq · sqrt( (1 + b) / ((1 − b)·n) )
n_efectivo   = n·(1 − b)/(1 + b)
```

**Ejemplo (prueba de recuperación).** `λ = 0,05`, `b = e^{−0,05} = 0,9512`,
`n = 5.000`:

```text
n_efectivo = 5.000 · 0,0488 / 1,9512 ≈ 125
EE(μ̂)     = σ_eq / sqrt(125) ≈ 0,089·σ_eq
```

5.000 observaciones equivalen a unas 125 independientes para estimar `μ`. Por
eso la prueba exige `|μ̂ − μ| ≤ 3·EE(μ̂)` en lugar de un error relativo (con
`μ = 0`, el error relativo ni siquiera está definido). Con la semilla 42 el error
queda en torno a `0,13·σ_eq` (≈ 1,4 EE): un resultado normal. Exigir
`|μ̂| < 0,10·σ_eq` (≈ 1,12 EE) fallaría en alrededor de una de cada cuatro
semillas sin que el estimador tenga error alguno.

**La misma lógica para `λ`.** `EE(b̂) ≈ sqrt((1 − b²)/n)` y
`EE(λ̂) ≈ EE(b̂)/b`. Con `λ = 0,05`:

| n | Error relativo típico de `λ̂` | Vida media con ± 2 EE |
|---|---|---|
| 5.000 | ≈ 9 % | 11,7 a 17,0 días |
| 1.000 | ≈ 20 % | 9,8 a 23,5 días |
| 252 | ≈ 41 % | 7,6 a 75,8 días |

`λ` es el parámetro más difícil de estimar. Esto explica por qué varía tanto entre
ventanas de 252 observaciones y por qué las bandas que ignoran la incertidumbre de
parámetros son demasiado estrechas (sección 9).

### 8.7 Diagnóstico: residuos y estabilidad

Si el AR(1) captura la dinámica del spread, sus residuos
`ε̂_t = X_{t+1} − â − b̂·X_t` deben parecer ruido sin memoria.

```text
ACF:        ρ̂_k para k = 1..20,  banda aproximada ± 1,96/√n
Ljung–Box:  Q(m) = n·(n + 2)·Σ_{k=1..m} ρ̂_k² / (n − k)  ~  χ² con m − 1 grados de libertad
```

- Se resta 1 grado de libertad porque se estimó un coeficiente AR.
- Se calcula con `m = 10` y `m = 20`. Si `p(20) < 0,05` se detecta
  autocorrelación: el AR(1)/OU no captura toda la dinámica, y las bandas, los
  intervalos y la vida media pueden ser poco fiables.
- No detectar autocorrelación es compatible con el modelo, pero no prueba que sea
  correcto.

**Estabilidad del OU (decisión E8).** Se ajusta el AR(1) en las mismas ventanas
de la sección 6 (252 observaciones, paso 63) sobre el spread de formación con `α`
y `β` **fijos** (el spread que realmente se opera). En cada ventana con
`0 < b̂ < 1` se calculan `λ`, `μ`, `σ` y la vida media, **sin exigir** que la
prueba de raíz unitaria rechace:

```text
CV(λ) = desviación estándar(λ_ventanas) / media(λ_ventanas)     (ddof = 1, ventanas con 0 < b̂ < 1)
```

Cuanto mayor es el CV, menos estable es la velocidad de reversión. Con pocas
observaciones por ventana es esperable cierta variación (ver la tabla de la
sección 8.6).

La prueba ADF de cada ventana se informa **aparte** (rechaza o no, estadístico,
rezagos y p-value), con los valores críticos de una serie observada (N = 1): la
ventana es un tramo de un spread cuyos `α` y `β` se estimaron con toda la
formación, no en la ventana, así que la corrección N = 2 no corresponde.

¿Por qué no filtrar las ventanas con la prueba? Porque con 252 observaciones tiene
poca potencia: en un OU verdadero muchas ventanas no rechazan, y las que sí lo
hacen son las de `λ` más alto. Quedarse solo con ellas haría que la vida media
pareciera más corta y el CV más pequeño de lo que son. Simulación (pares
cointegrados de 1.500 observaciones; ventanas de 252 sobre los 900 datos de
formación; 100 pares por fila, 400 en la de paseos):

| Spread | Ventanas con parámetros | Mediana de la vida media en las ventanas | La prueba ADF rechaza en |
|---|---|---|---|
| OU con `λ = 0,05` (vida media 13,9 días) | 100 % | 10,8 días | 48 % de las ventanas |
| OU con `λ = 0,1` (vida media 6,9 días) | 100 % | 6,1 días | 93 % de las ventanas |
| Paseos aleatorios independientes (no hay OU) | — | — | 7,2 % de las ventanas |

Lectura: con un OU lento, la mitad de las ventanas «no rechaza» aunque el spread
revierta, así que no rechazar en una ventana no prueba que la dinámica haya
cambiado. La vida media de las ventanas sale algo más corta que la verdadera
porque el AR(1) sobrestima `λ` en muestras cortas. Si se exigiera el rechazo con
los valores críticos N = 2 (regla anterior), con `λ = 0,05` solo el 17 % de las
ventanas tenía parámetros, la mediana bajaba a 6,6 días y 48 de los 100 pares se
quedaban sin CV(λ). Con paseos
independientes la prueba rechaza algo más del 5 % nominal porque `α` y `β` se
estimaron con toda la formación y por la selección de rezagos en muestras cortas.
La validez del OU la decide siempre el ajuste de toda la formación (sección 8.5).

**En el código:** `mrlab.model.ou` (`fit_ou`, `adf_unit_root_test`,
`conditional_mean`, `conditional_variance`, `forecast`, `simulate_ou_exact`,
`simulate_paths_summary`), `mrlab.model.diagnostics` (`residual_diagnostics`,
`rolling_ou_stability`).

---

## 9. Bandas, intervalos e incertidumbre de parámetros

Son tres cosas distintas que se suelen confundir:

| Objeto | Fórmula | Responde a |
|---|---|---|
| **Bandas estacionarias** | `μ ± k·σ_eq` (k = 1, 2) | ¿Dónde suele moverse el spread a largo plazo? |
| **Intervalo predictivo condicional** | `E[X_{t+h} \| X_t] ± 1,96·sqrt(Var[X_{t+h} \| X_t])` | Partiendo del valor de hoy, ¿dónde podría estar dentro de `h` días? |
| **Incertidumbre de parámetros** | No incluida en la v1 | ¿Cuánto me equivoco al estimar `λ`, `μ` y `σ`? |

- Los intervalos condicionales (al 95 %, normales, decisión O3) son **más
  estrechos** a horizontes cortos y se acercan a la banda estacionaria cuando `h`
  crece. Con `λ = 0,05`, la desviación condicional es el 31 % de `σ_eq` a 1 día,
  el 63 % a 5 días y el 93 % a 20 días.
- Bandas e intervalos usan `λ̂`, `μ̂` y `σ̂` **como si fueran exactos**. Como no
  lo son (sección 8.6), ambos son más estrechos de lo que deberían. La interfaz lo
  avisa. El bootstrap por bloques queda como extensión.
- Los umbrales de z ± 2 de la estrategia equivalen a las bandas estacionarias
  `μ ± 2·σ_eq`.

Pronósticos que muestra el laboratorio: `h = 1, 5 y 20` días, desde el último dato
de formación.

**En el código:** `mrlab.model.ou.forecast`.

---

## 10. z-score fijo y z-score móvil

**z fijo (el que usa la estrategia):**

```text
z_t = (X_t − μ̂) / σ̂_eq        con μ̂ y σ̂_eq FIJOS, estimados en formación
```

Dice cuántas desviaciones estacionarias separan al spread de su referencia
estadística. Como los parámetros son de formación, en validación y prueba no usa
información futura ni necesita periodo de calentamiento.

**z móvil (solo comparación pedagógica):**

```text
z_t^móvil = (X_t − m_t) / s_t
m_t, s_t = media y desviación estándar (ddof = 1) de las últimas w observaciones, hasta t inclusive
```

Con `w = 63` por defecto. No usa información futura, pero tiene tres problemas:

1. **Persigue al spread.** Si la relación se rompe y el spread salta a un nivel
   nuevo, la media móvil lo alcanza en `w` observaciones y el z vuelve a 0. Parece
   una reversión que en los precios nunca ocurrió. El z fijo, en cambio, se queda
   en valores extremos y deja ver el problema.
2. Su escala `s_t` no es la `σ_eq` del modelo: los umbrales pierden su
   interpretación en términos del OU.
3. Añade un parámetro más (`w`) que ajustar, con más riesgo de sobreajuste.

**En el código:** `mrlab.model.ou.zscore`, `mrlab.model.ou.rolling_zscore`.

---

## 11. Reglas de la estrategia

### 11.1 Parámetros por defecto

| Parámetro | Valor por defecto |
|---|---|
| Entrada (z) | ± 2,0 |
| Salida (z) | ± 0,5 |
| Stop-loss (z) | ± 3,5 |
| Stop-loss monetario | 5 % del capital inicial asignado al par |
| Tiempo máximo | 3 × vida media, redondeado, mínimo 5 observaciones |
| Enfriamiento tras stop | 5 observaciones |
| Capital inicial | 10.000 USD |
| Exposición bruta objetivo | 100 % del capital |
| Apalancamiento máximo | 2× |
| Ventas en corto | Permitidas (configurable) |

Requisito: `salida < entrada < stop`. La estrategia solo existe si el OU es
válido (decisiones O5 y O7) y no hay ninguna razón que bloquee la simulación
(series colineales, índice o futuro sin instrumento ejecutable, precios
ejecutables ≤ 0). Si el par no está en ✅ o tiene alguna razón de descarte, el
backtest se permite con un aviso permanente de «resultado solo ilustrativo» que
cita los motivos (decisión O5a).

### 11.2 Entradas

```text
z_t > +entrada  →  VENDER el spread   (corto A; largo B si β > 0)
z_t < −entrada  →  COMPRAR el spread  (largo A; corto B si β > 0)
```

No hay entradas con posición abierta, en estado CERRANDO, durante el
ENFRIAMIENTO ni en un lado bloqueado por el rearme.

### 11.3 Salidas y su prioridad

| Condición | Corto del spread | Largo del spread |
|---|---|---|
| Salida normal (decisión S1) | `z ≤ +salida` | `z ≥ −salida` |
| Stop por z | `z ≥ +stop` | `z ≤ −stop` |
| Stop monetario (decisión S6) | PnL neto acumulado de la operación ≤ −5 % × capital inicial | igual |
| Tiempo máximo | observaciones desde la ejecución de la entrada ≥ máximo | igual |

- **Salidas direccionales (S1).** Leer «|z| ≤ salida» literalmente dejaría
  abierta una posición corta si z salta de +1 a −1, es decir, si el spread revirtió
  **más allá** de `μ` a tu favor.
- **Prioridad si coinciden varias:** stop monetario > stop por z > tiempo máximo >
  salida normal. Se registran **todas** las condiciones activas y la aplicada.
- **Stop monetario:** con 10.000 USD de capital, se activa cuando la operación
  pierde 500 USD netos (costes incluidos). Se evalúa al cierre `t` y se ejecuta en
  `t+1`, así que la pérdida real puede superar el 5 % si hay un salto de precio.
- **Tiempo máximo:** con vida media de 13,9 días, `3 × 13,9 = 41,6` → 42
  observaciones.

### 11.4 Enfriamiento y rearme

- **Enfriamiento (S3), solo tras un stop:** si la salida se ejecuta en la
  observación `e`, las señales de entrada de los cierres `e..e+4` se ignoran. La
  primera entrada posible se decide en `e+5` y se ejecuta en `e+6`: hay 5
  observaciones completas sin posición.
- **Rearme (S2), tras un stop o el tiempo máximo:** no se vuelve a entrar **por
  ese lado** hasta que z regrese dentro de ± entrada en algún cierre posterior. El
  lado contrario no se bloquea. Sin esta regla, un stop en z = 3,6 seguido de
  z = 3,0 al terminar el enfriamiento repetiría la misma operación perdedora.

Estados de la máquina: `SIN_POSICION`, `LARGO_SPREAD`, `CORTO_SPREAD`,
`CERRANDO`, `ENFRIAMIENTO`.

**En el código:** `mrlab.strategy.params`, `mrlab.strategy.state_machine`.

---

## 12. Dimensionamiento de las piernas

### 12.1 Spread logarítmico: β relaciona nocionales

```text
G = exposición_bruta_objetivo × capital      (acotada por apalancamiento_máximo × capital)
N = G / (1 + |β|)                             (nocional de la pierna A)

Comprar el spread:   q_A = +N / P_A ;   q_B = −β·N / P_B
Vender el spread:    q_A = −N / P_A ;   q_B = +β·N / P_B

Exposición bruta = |q_A·P_A| + |q_B·P_B| = N·(1 + |β|) = G
```

**Por qué nocionales.** Para movimientos pequeños,
`ΔX = Δln P_A − β·Δln P_B ≈ r_A − β·r_B`. Una posición con `N` USD en A y
`−β·N` USD en B gana aproximadamente `N·r_A − β·N·r_B = N·ΔX`: es proporcional al
cambio del spread. Esta aproximación solo explica el diseño; el PnL real se
calcula siempre con las cantidades (sección 14).

**Ejemplo.** Capital 10.000 USD, `G = 10.000`, `β = 0,8`, `P_A = 60`,
`P_B = 170`, comprar el spread:

| | Cálculo | Resultado |
|---|---|---|
| `N` | `10.000 / 1,8` | 5.555,56 USD |
| `q_A` | `+5.555,56 / 60` | +92,59 acciones (+5.555,56 USD) |
| `q_B` | `−0,8 · 5.555,56 / 170` | −26,14 acciones (−4.444,44 USD) |
| Exposición bruta | `5.555,56 + 4.444,44` | 10.000 USD |
| Exposición neta | `5.555,56 − 4.444,44` | 1.111,11 USD |

### 12.2 Spread en precios: β relaciona unidades

```text
q_A = ± G / (P_A + |β|·P_B) ;    q_B = ∓ β·q_A
```

Con cantidades fijas y `q_B = −β·q_A`, el PnL bruto es exactamente
`q_A·ΔX = q_A·ΔP_A − β·q_A·ΔP_B`.

**Ejemplo.** `β = 0,35`, `P_A = 60`, `P_B = 170`, comprar el spread:
`q_A = 10.000 / (60 + 0,35·170) = 83,68` acciones; `q_B = −0,35·83,68 = −29,29`
acciones. Exposición bruta: `83,68·60 + 29,29·170 ≈ 10.000` USD.

### 12.3 β negativo y operación sin cortos

- Con `β < 0`, `q_B = −β·(…)` tiene **el mismo signo** que `q_A`: comprar el
  spread es comprar A y comprar B; venderlo es vender ambos. Significa que, en la
  relación estimada, los dos activos se mueven en sentidos opuestos.
- **Sin ventas en corto (decisión S9):** con `β > 0` ninguna señal es ejecutable
  (siempre hay una pierna corta); con `β < 0`, comprar el spread (dos largos) sí lo
  es y venderlo no. Las señales no ejecutables se registran como «señal no
  ejecutable». No se opera una sola pierna: eso no es una estrategia de pares. En
  cripto spot, el corto real requiere derivados, fuera de la v1.

### 12.4 Capital, apalancamiento y redondeo

- **Decisión S5:** el nocional objetivo se calcula con el capital del cierre de la
  señal (`t`); las cantidades se obtienen con el precio de ejecución (cierre
  `t+1`), como una orden al cierre por nocional.
- **Decisión S7:** el límite de 2× se verifica en la entrada (si se supera, se
  reduce y se registra). Si durante la operación la exposición bruta supera
  2 × capital, se reduce proporcionalmente en la observación siguiente, con
  costes.
- Opción de unidades enteras: ambas cantidades se redondean **hacia cero** (nunca
  superan los límites). Si una pierna queda en 0, no se opera.

**En el código:** `mrlab.strategy.sizing`.

---

## 13. División de la muestra y convención de ejecución

### 13.1 División cronológica 60/20/20

| Segmento | Para qué se usa |
|---|---|
| Formación (60 %) | Selección del par, `α`, `β`, parámetros OU. Su backtest es solo ilustrativo (dentro de muestra) |
| Validación (20 %) | Ajustar umbrales de la estrategia |
| Prueba (20 %) | Se ejecuta **una vez** con la configuración congelada |

- `n_formación = floor(0,6·n)`, `n_validación = floor(0,2·n)`, prueba = resto.
  Las fechas de corte se congelan en el `RunConfig` (decisión S8).
- Para tener las 500 observaciones mínimas en formación se necesitan al menos
  834 observaciones comunes (`floor(0,6 · 834) = 500`).
- Cada segmento se simula por separado, empezando sin posición y con el capital
  inicial. Una posición abierta al final queda como no realizada.
- Si se cambian los umbrales después de ver la prueba, el resultado queda marcado
  como «prueba contaminada» de forma permanente (decisión S10). En la fase 1, el
  historial de configuraciones con que se ejecutó la prueba viaja dentro de
  `run_config.json` (`test_history`, decisión S10a); el registro persistente de
  estudios llega en la fase 2.

### 13.2 Convención de ejecución t → t+1

```text
Cierre de t:    se calcula z_t con información hasta t y se DECIDE la orden
Cierre de t+1:  se EJECUTA la orden al precio de cierre de t+1
```

| Día | Qué ocurre |
|---|---|
| Lunes (t) | Cierre: z = 2,15 > 2 → se decide vender el spread. El nocional se fija con el capital del lunes |
| Martes (t+1) | Se ejecuta al cierre del martes; las cantidades se calculan con los precios del martes |
| Miércoles | El PnL de la posición empieza a acumularse (precio del miércoles − precio del martes) |

**Por qué.** Para calcular `z_t` necesitas el cierre de `t`, y cuando lo conoces
la subasta de cierre ya terminó. Operar al mismo cierre usaría información que en
la práctica no tenías. El precio de esta prudencia es que se pierde (o se gana)
el movimiento entre `t` y `t+1`. Una prueba automática verifica que modificar los
precios posteriores a `t` no cambia señales, posiciones ni PnL hasta `t`.

**En el código:** `mrlab.backtest.split`, `mrlab.backtest.engine`.

---

## 14. PnL: ganancias y pérdidas en dinero

### 14.1 La fórmula

```text
ΔΠ_t = q_A,t−1·(P_A,t − P_A,t−1) + q_B,t−1·(P_B,t − P_B,t−1) − C_t
Capital_t = Capital_{t−1} + ΔΠ_t
```

- Se marcan a mercado las cantidades que había **al cierre anterior**. Lo que se
  ejecuta en el cierre `t` empieza a ganar o perder desde `t+1`.
- `C_t` incluye los costes de transacción ejecutados en `t` y los de
  mantenimiento del intervalo `(t−1, t]` (sección 15).
- `α` no interviene en el PnL: solo desplaza el nivel del spread.

### 14.2 Por qué NUNCA es la diferencia del spread logarítmico

1. **Unidades:** `ΔX` está en log-puntos, no en dinero.
2. **Aproximación de primer orden:** `N·ΔX` solo describe el PnL de movimientos
   pequeños con nocionales constantes. Con cantidades fijas, los nocionales
   cambian a medida que se mueven los precios.
3. **Los rendimientos logarítmicos no se suman entre activos** (sección 1).

**Ejemplo.** `β = 1`, `N = 5.000`: `q_A = +100` acciones de A a 50 USD y
`q_B = −50` acciones de B a 100 USD.

| Movimiento | PnL real (con cantidades) | `N·ΔX` (incorrecto) | Error |
|---|---|---|---|
| A: 50 → 55 (+10 %); B: 100 → 108 (+8 %) | `100·5 − 50·8` = **+100,00 USD** | `5.000·(ln 1,10 − ln 1,08)` = 91,75 USD | 8 % |
| A: 50 → 65 (+30 %); B: 100 → 105 (+5 %) | `100·15 − 50·5` = **+1.250,00 USD** | `5.000·(ln 1,30 − ln 1,05)` = 1.067,87 USD | 15 % |

Además, `N·ΔX` ignora los costes, el redondeo y las reducciones por
apalancamiento. Por eso el motor calcula siempre `q·ΔP`.

### 14.3 Dividendos

Los precios de Yahoo están **ajustados por splits y dividendos**
(`auto_adjust=True`): el dividendo ya aparece como rendimiento del precio
ajustado. Sumarlo aparte lo contaría dos veces.

### 14.4 Identidades contables (se verifican con pruebas)

```text
Σ PnL bruto − Σ costes = Σ PnL neto
Capital final = capital inicial + Σ PnL neto
Realizado + no realizado = PnL total
```

- **Realizado:** PnL neto de las operaciones cerradas.
- **No realizado:** PnL bruto marcado a mercado de la operación abierta menos los
  costes que ya generó. Todos los costes se atribuyen a una operación (decisión
  C3).

**En el código:** `mrlab.backtest.engine`.

---

## 15. Costes

### 15.1 Fórmulas

```text
Comisión (por pierna)      = |Δq|·P_ejecución · 5 / 10.000
Slippage (por pierna)      = |Δq|·P_ejecución · s / 10.000      (s = 5 pb acciones y ETF; 10 pb cripto)
Préstamo de cortos         = 3 %·|nocional corto_{t−1}|·d_t / 365
Financiación               = 5 %·max(0, exposición bruta_{t−1} − capital_{t−1})·d_t / 365
C_t                        = suma de los anteriores
d_t                        = días NATURALES entre la fecha anterior y la actual
```

- Comisión y slippage se cobran sobre el nocional operado de **cada pierna** y en
  **cada ajuste** de cantidades (entrada, salida, reducción por apalancamiento).
  El slippage se contabiliza como coste, sin alterar el precio de ejecución
  (decisión C1).
- Préstamo y financiación se devengan por **días naturales / 365**: el lunes
  devenga el fin de semana (decisión C2). Los montos se miden al cierre anterior.
- El efectivo no se remunera (coherente con la tasa libre de riesgo 0 del
  Sharpe).

### 15.2 Ejemplos

| Concepto | Cálculo | Monto |
|---|---|---|
| Entrada con `G = 10.000` (acciones) | `10.000·(5 + 5)/10.000` | 10,00 USD |
| Ida y vuelta (entrada + salida) | ≈ 2 × 10 | ≈ 20 USD (≈ 0,2 % del capital) |
| Préstamo de un corto de 4.444,44 USD durante 10 días naturales | `0,03·4.444,44·10/365` | 3,65 USD |
| Préstamo de viernes a lunes (3 días naturales) | `0,03·4.444,44·3/365` | 1,10 USD |
| Financiación con exposición 15.000 y capital 10.000, un día | `0,05·5.000/365` | 0,68 USD |

### 15.3 Costes frente al tamaño del movimiento

Como orden de magnitud (no es como se calcula el PnL), una operación de z = 2 a
z = 0,5 captura cerca de `N·(z_entrada − z_salida)·σ_eq`:

| `σ_eq` del spread | Movimiento bruto aproximado (N = 5.555,56) | Frente a ≈ 20 USD de costes |
|---|---|---|
| 3 % | `5.555,56·1,5·0,03` ≈ 250 USD | Los costes son ≈ 8 % |
| 0,5 % | `5.555,56·1,5·0,005` ≈ 42 USD | Los costes se comen ≈ la mitad |

Un spread con `σ_eq` muy pequeña puede revertir de forma impecable y aun así no
cubrir los costes.

**En el código:** `mrlab.backtest.costs`, `mrlab.strategy.params.CostParams`.

---

## 16. Métricas

### 16.1 Fórmulas

```text
r_t = Capital_t / Capital_{t−1} − 1               (rendimiento diario neto; incluye días sin posición)

Retorno total        R = Capital_final / Capital_inicial − 1
Retorno anualizado     = (1 + R)^(P / n_r) − 1     (n_r = número de rendimientos)
Drawdown_t             = Capital_t / max_{s ≤ t} Capital_s − 1
Máximo drawdown        = min_t Drawdown_t          (fracción ≤ 0)
Sharpe                 = media(r) / desv(r) · sqrt(P)      (desv con ddof = 1; rf = 0)
% de aciertos          = operaciones cerradas con PnL neto > 0 / operaciones cerradas
Factor de beneficio    = Σ ganancias / |Σ pérdidas|        (operaciones cerradas)
Rotación               = Σ nocional operado / capital inicial;   anualizada = rotación · P / n_r
Costes / PnL bruto     = costes totales / PnL bruto
```

- `P` = **252** para acciones, ETF y pares mixtos; **365** solo para pares de dos
  criptomonedas. La convención aparece junto al número («rf = 0; anualizado con
  252»). No se puede cambiar `P` a voluntad: debe coincidir con el número de
  observaciones por año del calendario usado.
- **Duración del drawdown:** observaciones desde el pico hasta la recuperación (o
  hasta el final, si no se recupera).
- **Casos no definidos (decisión C4):** Sharpe si la desviación es 0 o hay menos
  de 2 rendimientos; factor de beneficio si no hay pérdidas («no definido»);
  costes como % del bruto si el PnL bruto ≤ 0 («no aplica»).
- Las operaciones abiertas al final del segmento se cuentan aparte.
- También se informan: ganancia media, pérdida media, duración media de las
  operaciones, exposición bruta media / capital y apalancamiento máximo usado.
- **Avisos:** con menos de 10 operaciones, «muestra insuficiente para conclusiones
  estadísticas». Si los costes superan un PnL bruto positivo, se avisa de que los
  costes se comen el resultado.

### 16.2 Ejemplos

- **Sharpe:** media diaria 0,04 %, desviación diaria 0,8 % →
  `0,0004/0,008·sqrt(252) = 0,05·15,87 ≈ 0,79`.
- **Retorno anualizado:** `R = 10 %` en 504 rendimientos (≈ 2 años) →
  `1,10^(252/504) − 1 = 4,88 %` anual.
- **Drawdown:** el capital sube de 10.000 a 10.800 (pico) y baja a 9.900 →
  `9.900/10.800 − 1 = −8,33 %`.
- **Operaciones:** ganancias de 600, 400 y 300; pérdidas de −200 y −450 →
  aciertos 60 %, ganancia media 433,33, pérdida media −325, factor de beneficio
  `1.300/650 = 2,0`. Con solo 5 operaciones aparece el aviso de muestra
  insuficiente.
- **Rotación:** cada ida y vuelta con `G = 10.000` opera ≈ 20.000 USD de nocional,
  es decir, una rotación de 2,0 por operación.

### 16.3 Para tener en cuenta

Un Sharpe es una **estimación**, con error. Con rendimientos aproximadamente
independientes, el error estándar del Sharpe anualizado es del orden de
`1/sqrt(años)` (Lo, 2002). En una validación de 2 años, un Sharpe de 0,79 tiene un
error estándar de ≈ 0,7: no se distingue de cero. Por eso el laboratorio
insiste en el tamaño de la muestra y nunca ordena pares por resultados
retrospectivos.

**En el código:** `mrlab.backtest.metrics`.

---

## 17. Mapa de fórmulas y código

| Tema | Sección | Módulo |
|---|---|---|
| Rendimientos y correlaciones | 1, 2 | `mrlab/stats/correlation.py` |
| ADF, KPSS e interpretación combinada | 3 | `mrlab/stats/integration.py` |
| Engle–Granger, dirección y decisión | 4, 5.4 | `mrlab/stats/cointegration.py` |
| Benjamini–Hochberg / Yekutieli | 5 | `mrlab/stats/multitest.py` |
| Ventanas de estabilidad | 6, 8.7 | `mrlab/stats/windows.py`, `mrlab/stats/cointegration.py`, `mrlab/model/diagnostics.py` |
| Spread, α, β | 7 | `mrlab/model/spread.py` |
| OU: estimación, pronóstico, simulación, z | 8, 9, 10 | `mrlab/model/ou.py` |
| ACF, Ljung–Box | 8.7 | `mrlab/model/diagnostics.py` |
| Parámetros, estados, dimensionamiento | 11, 12 | `mrlab/strategy/` |
| División, motor, costes, métricas | 13–16 | `mrlab/backtest/` |
| Estado del par y causas | 6.3 | `mrlab/analysis/pair_pipeline.py` |
| Escenarios sintéticos | 2.2 | `mrlab/data/providers/synthetic.py` |

---

## 18. Referencias

- Benjamini, Y. y Hochberg, Y. (1995). Controlling the false discovery rate: a
  practical and powerful approach to multiple testing. *Journal of the Royal
  Statistical Society, Series B*, 57(1), 289–300.
- Benjamini, Y. y Yekutieli, D. (2001). The control of the false discovery rate in
  multiple testing under dependency. *The Annals of Statistics*, 29(4),
  1165–1188.
- Dickey, D. A. y Fuller, W. A. (1979). Distribution of the estimators for
  autoregressive time series with a unit root. *Journal of the American
  Statistical Association*, 74(366), 427–431.
- Engle, R. F. y Granger, C. W. J. (1987). Co-integration and error correction:
  representation, estimation, and testing. *Econometrica*, 55(2), 251–276.
- Granger, C. W. J. y Newbold, P. (1974). Spurious regressions in econometrics.
  *Journal of Econometrics*, 2(2), 111–120.
- Hobijn, B., Franses, P. H. y Ooms, M. (2004). Generalizations of the KPSS-test
  for stationarity. *Statistica Neerlandica*, 58(4), 483–502.
- Kwiatkowski, D., Phillips, P. C. B., Schmidt, P. y Shin, Y. (1992). Testing the
  null hypothesis of stationarity against the alternative of a unit root.
  *Journal of Econometrics*, 54(1–3), 159–178.
- Ljung, G. M. y Box, G. E. P. (1978). On a measure of lack of fit in time series
  models. *Biometrika*, 65(2), 297–303.
- Lo, A. W. (2002). The statistics of Sharpe ratios. *Financial Analysts
  Journal*, 58(4), 36–52.
- MacKinnon, J. G. (1994). Approximate asymptotic distribution functions for
  unit-root and cointegration tests. *Journal of Business & Economic
  Statistics*, 12(2), 167–176.
- MacKinnon, J. G. (2010). Critical values for cointegration tests. Queen's
  Economics Department Working Paper No. 1227.
- Said, S. E. y Dickey, D. A. (1984). Testing for unit roots in
  autoregressive-moving average models of unknown order. *Biometrika*, 71(3),
  599–607.
- Stock, J. H. (1987). Asymptotic properties of least squares estimators of
  cointegrating vectors. *Econometrica*, 55(5), 1035–1056.
- Uhlenbeck, G. E. y Ornstein, L. S. (1930). On the theory of the Brownian
  motion. *Physical Review*, 36(5), 823–841.
