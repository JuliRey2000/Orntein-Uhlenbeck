# Datos de ejemplo — SINTÉTICOS

> **Estos archivos contienen datos SINTÉTICOS generados por el laboratorio.
> NO son precios históricos de ningún activo real.**

> Herramienta educativa. No constituye asesoría de inversión. Resultados
> históricos y simulados no garantizan resultados futuros.

| Archivo | Contenido |
|---|---|
| `SINTETICO_SINT-COI-A.csv` | Serie A del par cointegrado sintético (`date,close`) |
| `SINTETICO_SINT-COI-A.meta.json` | Metadatos de la serie A: moneda, clase de activo, tipo de precio y origen `SINTETICO` |
| `SINTETICO_SINT-COI-B.csv` | Serie B del par cointegrado sintético (`date,close`) |
| `SINTETICO_SINT-COI-B.meta.json` | Metadatos de la serie B |

## Cómo se generaron

Con el generador sintético del laboratorio (`SyntheticProvider`, escenario
«cointegrado», semilla 7), 1.500 días hábiles (lunes a viernes, sin festivos)
desde el 2018-01-01:

- `log B` es un paseo aleatorio (deriva 0,0002 y volatilidad 0,01 por día).
- `log A = log B + X`, donde `X` es un proceso de Ornstein–Uhlenbeck simulado con
  su transición exacta: λ = 0,1 por día (vida media ≈ 6,9 días), μ = 0 y
  σ = 0,01.

Como el spread se construyó así, **sabemos de antemano** que el par está
cointegrado: sirve para practicar el flujo completo y comprobar que el
laboratorio lo detecta. Con datos reales nunca se sabe de antemano, y eso es
justamente lo que el laboratorio ayuda a evaluar.

Para regenerarlos (el resultado es idéntico byte a byte):

```bash
python examples/generar_csv_sinteticos.py
```

## Cómo se usan

```bash
python examples/demo_ko_pep.py
```

Sin argumentos, la demo analiza este par sintético. Todo lo que exporta lleva la
marca SINTÉTICO: el prefijo `SINTETICO_` en los nombres de archivo y la columna
`tipo_datos = SINTETICO` en cada fila.

## Datos reales

`make fetch-example` (o `python examples/fetch_ko_pep.py`) descarga KO y PEP de
Yahoo Finance a `examples/data/real/`, **solo en tu máquina**. Esa carpeta no se
versiona: los términos de uso de Yahoo desaconsejan redistribuir sus datos.
