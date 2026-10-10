# Datos de ejemplo — SINTÉTICOS

> **Estos archivos contienen datos SINTÉTICOS generados por el laboratorio.
> NO son precios históricos de ningún activo real.**

> Herramienta educativa. No constituye asesoría de inversión. Resultados
> históricos y simulados no garantizan resultados futuros.

| Archivo | Contenido |
|---|---|
| `SINTETICO_SINT-COI-A.csv` | Serie A del par cointegrado sintético (`date,close,tipo_datos`) |
| `SINTETICO_SINT-COI-A.meta.json` | Metadatos de la serie A: moneda, clase de activo, tipo de precio, origen `SINTETICO` y semilla (`seed`) |
| `SINTETICO_SINT-COI-B.csv` | Serie B del par cointegrado sintético (`date,close,tipo_datos`) |
| `SINTETICO_SINT-COI-B.meta.json` | Metadatos de la serie B |

La marca SINTÉTICO viaja con los datos por tres vías independientes: la columna
`tipo_datos = SINTETICO` en cada fila, el prefijo `SINTETICO_` del nombre del
archivo y los metadatos. Si copias solo un CSV (sin su `.meta.json`) y lo declaras
como histórico, el laboratorio lo sigue tratando como SINTÉTICO y lo avisa.

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

Para regenerarlos (el resultado es idéntico byte a byte; una prueba automática lo
comprueba):

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

La muestra de prueba solo se ejecuta con `--incluir-prueba`. La marca «Prueba
contaminada» usa el historial guardado en el `run_config.json` exportado: una
nueva ejecución en la misma carpeta lo conserva (`--reiniciar-historial` lo
descarta a conciencia). Limitación de la fase 1: si exportas en otra carpeta
(`--output-dir`) o partes de un `run_config.json` copiado antes de ejecutar la
prueba, ese historial no viaja con él y la marca no puede detectarlo; el registro
persistente de estudios llega en la fase 2.

## Datos reales

`make fetch-example` (o `python examples/fetch_ko_pep.py`) descarga KO y PEP de
Yahoo Finance a `examples/data/real/`, **solo en tu máquina**. Esa carpeta no se
versiona: los términos de uso de Yahoo desaconsejan redistribuir sus datos.
