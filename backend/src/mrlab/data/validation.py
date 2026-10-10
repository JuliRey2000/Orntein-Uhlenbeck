"""Validación de datos previa a cualquier análisis (sección 2.3) y alineación (2.4).

CONTRATO:
- ``clean_series(series)`` → (serie limpia, SeriesValidation):
  * desorden temporal → se ordena y se reporta (DESORDEN_TEMPORAL, advertencia);
  * fechas duplicadas → se conserva la ÚLTIMA aparición, aviso con las fechas
    (DUPLICADOS, advertencia);
  * close NaN → se ELIMINA la fila (nunca forward-fill), se reporta con fechas
    (FALTANTES, advertencia);
  * close ≤ 0 → se reporta (PRECIO_NO_POSITIVO, error) y NO se elimina: la
    decisión de bloquear la toma ``validate_pair`` según el tipo de spread;
  * |ln(P_t/P_{t−1})| > ln(1,5) → advertencia (SALTO_SOSPECHOSO) con fechas:
    posible error de ajuste. El umbral es simétrico en logaritmos: una subida de
    más del 50 % o una caída de más del 33,3 % (así un split 2:1 sin ajustar,
    ≈ −50 %, siempre se marca);
  * cripto (calendario 7 días): huecos de más de 1 día natural → advertencia
    (HUECO_CRIPTO) con las fechas que faltan. Para bolsa no se infieren festivos.
- ``align_pair(a, b)`` → AlignedPair: intersección de fechas de las series
  limpias (si recibe series crudas las limpia antes); NUNCA se rellenan fechas.
  Registra las fechas que solo estaban en una de las dos series. Calendario
  según ``mrlab.data.calendar.calendar_type``.
- ``validate_pair(a, b, spread_type, min_train_obs=500, train_fraction=0.6)``
  → (ValidationReport, AlignedPair | None):
  * monedas distintas → blocked=True, currency_mismatch=True, error MONEDA_DISTINTA
    explicando que la conversión de moneda queda como extensión de la v1;
    moneda desconocida (None) → blocked=True, MONEDA_DESCONOCIDA;
  * precios ≤ 0 en fechas comunes → log_spread_allowed=False; si spread_type es
    LOG → blocked=True (SPREAD_LOG_BLOQUEADO, explica que el log exige precios > 0);
    con spread en precios solo se advierte (PRECIOS_NO_POSITIVOS);
  * n_train = floor(n_alineadas · train_fraction) < min_train_obs →
    insufficient_data=True, blocked=True, error DATOS_INSUFICIENTES con números
    (cuántas hay, cuántas hacen falta, cuántas totales se necesitan);
  * fechas presentes en una sola serie → informativo (FECHAS_NO_COMUNES);
  * calendario mixto → advertencia DESFASE_CIERRE_CRIPTO (texto en calendar.py);
  * ``status_message``: resumen en una frase, en español. «Datos aptos…» solo
    si no hay ninguna incidencia con severidad error; si no está bloqueado pero
    hay errores (p. ej. precios ≤ 0 con spread en precios) → «Datos utilizables
    con incidencias: …» nombrando cada error.
  ``report.issues`` contiene TODAS las incidencias (primero las de cada serie,
  con ``symbol``, y después las del par): siempre se listan todas las causas.
  Si está bloqueado por moneda, se devuelve AlignedPair=None.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum

import numpy as np
import pandas as pd
from pydantic import BaseModel

from mrlab.data.calendar import CLOSE_TIME_MISMATCH_WARNING, calendar_type, periods_per_year
from mrlab.domain.errors import ConfigError, DataError
from mrlab.domain.models import AssetClass, CalendarType, PriceSeries, SpreadType, coverage_of

#: Subida diaria máxima antes de marcar un salto; el umbral real es simétrico en
#: logaritmos, |ln(P_t/P_{t−1})| > ln(1 + SUSPICIOUS_JUMP): +50 % o −33,3 %.
SUSPICIOUS_JUMP = 0.50
SUSPICIOUS_LOG_JUMP = math.log1p(SUSPICIOUS_JUMP)
SUSPICIOUS_DROP = math.expm1(-SUSPICIOUS_LOG_JUMP)  # −1/3
MIN_TRAIN_OBS = 500
MAX_DATES_IN_MESSAGE = 8


class Severity(StrEnum):
    INFO = "info"
    WARNING = "advertencia"
    ERROR = "error"


class ValidationIssue(BaseModel):
    code: str
    severity: Severity
    message: str
    symbol: str | None = None
    dates: list[date] = []
    details: dict = {}


class SeriesValidation(BaseModel):
    symbol: str
    n_raw: int
    n_clean: int
    unsorted: bool = False
    duplicates_removed: int = 0
    duplicate_dates: list[date] = []
    missing_values: int = 0
    missing_dates: list[date] = []
    non_positive: int = 0
    non_positive_dates: list[date] = []
    suspicious_jumps: list[date] = []
    crypto_gaps: list[date] = []
    issues: list[ValidationIssue] = []


class ValidationReport(BaseModel):
    series: list[SeriesValidation]
    calendar: CalendarType | None = None
    n_aligned: int = 0
    n_train_estimate: int = 0
    min_train_obs: int = MIN_TRAIN_OBS
    only_in_a: int = 0
    only_in_b: int = 0
    currency_mismatch: bool = False
    log_spread_allowed: bool = True
    insufficient_data: bool = False
    blocked: bool = False
    issues: list[ValidationIssue] = []
    status_message: str = ""


@dataclass(frozen=True)
class AlignedPair:
    prices: pd.DataFrame  # columnas "a", "b"; índice = fechas comunes (ordenadas, únicas)
    symbol_a: str
    symbol_b: str
    calendar: CalendarType
    only_in_a: list[date] = field(default_factory=list)
    only_in_b: list[date] = field(default_factory=list)


# --------------------------------------------------------------------------- formato


def format_dates(dates: Sequence[date], limit: int = MAX_DATES_IN_MESSAGE) -> str:
    """Lista de fechas ISO para un mensaje, recortada: «2020-01-02, 2020-01-03 y 5 más»."""
    shown = ", ".join(d.isoformat() for d in dates[:limit])
    rest = len(dates) - limit
    return f"{shown} y {rest} más" if rest > 0 else shown


def format_int(n: int) -> str:
    """Entero con separador de miles en español: 4214 → «4.214»."""
    return f"{n:,}".replace(",", ".")


def format_decimal(x: float, digits: int = 1) -> str:
    """Decimal con coma: 3.36 → «3,4»."""
    return f"{x:.{digits}f}".replace(".", ",")


def format_pct(fraction: float) -> str:
    """Porcentaje con espacio, como se escribe en español: 0.6 → «60 %»."""
    return f"{fraction * 100:.0f} %"


def _to_dates(idx: pd.Index) -> list[date]:
    return [ts.date() for ts in idx]


# --------------------------------------------------------------------------- una serie


def clean_series(series: PriceSeries) -> tuple[PriceSeries, SeriesValidation]:
    """Limpia una serie cruda y reporta todo lo que se cambió o se detectó."""
    symbol = series.symbol
    data = series.data
    if not isinstance(data.index, pd.DatetimeIndex):
        raise DataError(
            f"La serie «{symbol}» no está indexada por fechas.",
            code="INDICE_NO_FECHAS",
            details={"symbol": symbol},
        )
    if "close" not in data.columns:
        raise DataError(
            f"La serie «{symbol}» no tiene columna de precio de cierre («close»).",
            code="FALTA_CIERRE",
            details={"symbol": symbol, "columnas": [str(c) for c in data.columns]},
        )

    n_raw = len(data)
    out = data.copy()  # la serie limpia nunca comparte datos con la cruda
    issues: list[ValidationIssue] = []

    unsorted = not out.index.is_monotonic_increasing
    if unsorted:
        # Orden estable: entre fechas repetidas se conserva el orden original,
        # así «la última aparición» sigue siendo la última del archivo.
        out = out.sort_index(kind="stable")
        issues.append(
            ValidationIssue(
                code="DESORDEN_TEMPORAL",
                severity=Severity.WARNING,
                symbol=symbol,
                message=(
                    f"{symbol}: las fechas no estaban en orden cronológico. Se ordenaron "
                    "antes del análisis; revisa cómo se generó el archivo."
                ),
            )
        )

    dup_mask = out.index.duplicated(keep="last")
    duplicate_dates = sorted(set(_to_dates(out.index[dup_mask])))
    duplicates_removed = int(dup_mask.sum())
    if duplicates_removed:
        out = out[~dup_mask]
        issues.append(
            ValidationIssue(
                code="DUPLICADOS",
                severity=Severity.WARNING,
                symbol=symbol,
                dates=duplicate_dates,
                details={"filas_eliminadas": duplicates_removed},
                message=(
                    f"{symbol}: {len(duplicate_dates)} fecha(s) repetida(s) "
                    f"({format_dates(duplicate_dates)}). Se conservó el último valor de cada "
                    f"fecha y se eliminaron {duplicates_removed} fila(s)."
                ),
            )
        )

    finite = np.isfinite(out["close"].to_numpy(dtype=float))
    missing_dates = _to_dates(out.index[~finite])
    if missing_dates:
        out = out[finite]
        issues.append(
            ValidationIssue(
                code="FALTANTES",
                severity=Severity.WARNING,
                symbol=symbol,
                dates=missing_dates,
                message=(
                    f"{symbol}: {len(missing_dates)} fecha(s) sin precio de cierre "
                    f"({format_dates(missing_dates)}). Se eliminaron esas filas: no se rellenan "
                    "con el precio anterior (forward-fill) para no inventar observaciones."
                ),
            )
        )

    close = out["close"].astype(float)
    non_positive_dates = _to_dates(out.index[(close <= 0).to_numpy()])
    if non_positive_dates:
        issues.append(
            ValidationIssue(
                code="PRECIO_NO_POSITIVO",
                severity=Severity.ERROR,
                symbol=symbol,
                dates=non_positive_dates,
                message=(
                    f"{symbol}: {len(non_positive_dates)} precio(s) menores o iguales a cero "
                    f"({format_dates(non_positive_dates)}). El spread logarítmico exige precios "
                    "estrictamente positivos (no existe el logaritmo de 0 ni de un número negativo)."
                ),
            )
        )

    prev = close.shift(1)
    both_positive = ((close > 0) & (prev > 0)).to_numpy()
    log_change = np.full(len(close), np.nan)
    log_change[both_positive] = np.log(
        close.to_numpy()[both_positive] / prev.to_numpy()[both_positive]
    )
    jumps = both_positive & (np.abs(np.nan_to_num(log_change)) > SUSPICIOUS_LOG_JUMP)
    suspicious = _to_dates(out.index[jumps])
    if suspicious:
        issues.append(
            ValidationIssue(
                code="SALTO_SOSPECHOSO",
                severity=Severity.WARNING,
                symbol=symbol,
                dates=suspicious,
                details={"umbral_log": SUSPICIOUS_LOG_JUMP},
                message=(
                    f"{symbol}: salto de precio sospechoso en {len(suspicious)} fecha(s) "
                    f"({format_dates(suspicious)}): una subida de más del "
                    f"{format_pct(SUSPICIOUS_JUMP)} o una caída de más del "
                    f"{format_decimal(-SUSPICIOUS_DROP * 100)} % en un día "
                    "(|ln(Pₜ/Pₜ₋₁)| > ln 1,5). Puede ser un error de ajuste por split o "
                    "dividendo (un split 2:1 sin ajustar aparece como una caída del 50 %), o un "
                    "dato erróneo: revísalo antes de confiar en el análisis."
                ),
            )
        )

    crypto_gaps: list[date] = []
    if series.meta.asset.asset_class == AssetClass.CRYPTO and len(out) > 1:
        every_day = pd.date_range(out.index[0], out.index[-1], freq="D")
        crypto_gaps = _to_dates(every_day.difference(out.index))
        if crypto_gaps:
            issues.append(
                ValidationIssue(
                    code="HUECO_CRIPTO",
                    severity=Severity.WARNING,
                    symbol=symbol,
                    dates=crypto_gaps,
                    message=(
                        f"{symbol}: faltan {len(crypto_gaps)} día(s) en una serie de cripto, que "
                        f"cotiza los 7 días ({format_dates(crypto_gaps)}). No se rellenan: el "
                        "rendimiento posterior a cada hueco abarca varios días."
                    ),
                )
            )

    start, end, n_obs = coverage_of(out)
    cleaned = series.with_data(out).with_meta(coverage_start=start, coverage_end=end, n_obs=n_obs)
    report = SeriesValidation(
        symbol=symbol,
        n_raw=n_raw,
        n_clean=len(out),
        unsorted=unsorted,
        duplicates_removed=duplicates_removed,
        duplicate_dates=duplicate_dates,
        missing_values=len(missing_dates),
        missing_dates=missing_dates,
        non_positive=len(non_positive_dates),
        non_positive_dates=non_positive_dates,
        suspicious_jumps=suspicious,
        crypto_gaps=crypto_gaps,
        issues=issues,
    )
    return cleaned, report


# --------------------------------------------------------------------------- par


def _align_clean(a: PriceSeries, b: PriceSeries) -> AlignedPair:
    ia, ib = a.data.index, b.data.index
    common = ia.intersection(ib).sort_values()
    prices = pd.DataFrame(
        {
            "a": a.data["close"].reindex(common).astype(float),
            "b": b.data["close"].reindex(common).astype(float),
        },
        index=pd.DatetimeIndex(common, name="date"),
    )
    return AlignedPair(
        prices=prices,
        symbol_a=a.symbol,
        symbol_b=b.symbol,
        calendar=calendar_type([a.meta.asset.asset_class, b.meta.asset.asset_class]),
        only_in_a=_to_dates(ia.difference(ib).sort_values()),
        only_in_b=_to_dates(ib.difference(ia).sort_values()),
    )


def align_pair(a: PriceSeries, b: PriceSeries) -> AlignedPair:
    """Intersección de fechas de ambas series (limpias). Nunca rellena fechas."""
    clean_a, _ = clean_series(a)
    clean_b, _ = clean_series(b)
    return _align_clean(clean_a, clean_b)


def _currency(series: PriceSeries) -> str | None:
    cur = series.meta.asset.currency
    return cur.strip() if cur and cur.strip() else None


def _min_total_obs(min_train_obs: int, train_fraction: float) -> int:
    """Menor n con floor(n·fracción) ≥ mínimo (misma fórmula que ``compute_split``)."""
    n = math.ceil(min_train_obs / train_fraction)
    while math.floor(n * train_fraction) < min_train_obs:
        n += 1
    return n


#: Resumen breve de cada incidencia con severidad error que no bloquea el análisis.
_ERROR_SUMMARIES: dict[str, str] = {
    "PRECIO_NO_POSITIVO": "precios menores o iguales a cero",
}


def _usable_status(issues: list[ValidationIssue], n_aligned: int, n_train: int) -> str:
    """Resumen de un par no bloqueado: «Datos aptos…» solo si no hay errores."""
    counts = f"{format_int(n_aligned)} observaciones comunes, {format_int(n_train)} en formación"
    errors = [i for i in issues if i.severity == Severity.ERROR]
    n_warn = sum(i.severity == Severity.WARNING for i in issues)
    if not errors:
        base = f"Datos aptos para el análisis: {counts}"
        return base + (f", con {n_warn} advertencia(s) que conviene revisar." if n_warn else ".")
    described = []
    for issue in errors:
        if issue.code not in _ERROR_SUMMARIES:  # incidencia nueva: su primera frase
            described.append(issue.message.split(". ")[0].rstrip("."))
            continue
        where = f" en «{issue.symbol}»" if issue.symbol else ""
        dates = f" ({len(issue.dates)} fecha(s))" if issue.dates else ""
        described.append(f"{_ERROR_SUMMARIES[issue.code]}{where}{dates}")
    warn = f"; además hay {n_warn} advertencia(s)" if n_warn else ""
    return (
        f"Datos utilizables con incidencias: {'; '.join(described)}{warn}. {counts}. Revisa las "
        "incidencias antes de confiar en el análisis."
    )


def validate_pair(
    a: PriceSeries,
    b: PriceSeries,
    spread_type: SpreadType = SpreadType.LOG,
    min_train_obs: int = MIN_TRAIN_OBS,
    train_fraction: float = 0.6,
) -> tuple[ValidationReport, AlignedPair | None]:
    """Informe de validación del par y, si la moneda lo permite, las series alineadas."""
    if not 0 < train_fraction <= 1:
        raise ConfigError(
            "La fracción de formación debe estar entre 0 y 1.",
            code="DIVISION_INVALIDA",
            details={"train_fraction": train_fraction},
        )
    spread_type = SpreadType(spread_type)
    clean_a, val_a = clean_series(a)
    clean_b, val_b = clean_series(b)
    sym_a, sym_b = a.symbol, b.symbol
    issues: list[ValidationIssue] = [*val_a.issues, *val_b.issues]
    block_reasons: list[str] = []

    # Moneda
    cur_a, cur_b = _currency(a), _currency(b)
    unknown = [s for s, c in ((sym_a, cur_a), (sym_b, cur_b)) if c is None]
    currency_mismatch = False
    if unknown:
        names = " y ".join(f"«{s}»" for s in unknown)
        issues.append(
            ValidationIssue(
                code="MONEDA_DESCONOCIDA",
                severity=Severity.ERROR,
                details={"simbolos": unknown},
                message=(
                    f"No se conoce la moneda de {names}. Sin la moneda no se puede comprobar que "
                    "ambos activos coticen en la misma, así que el análisis queda bloqueado. Si "
                    "cargaste un CSV, declara su moneda; si no, prueba con otro proveedor."
                ),
            )
        )
        block_reasons.append(f"moneda desconocida para {names}")
    elif cur_a != cur_b:
        currency_mismatch = True
        issues.append(
            ValidationIssue(
                code="MONEDA_DISTINTA",
                severity=Severity.ERROR,
                details={"moneda_a": cur_a, "moneda_b": cur_b},
                message=(
                    f"«{sym_a}» cotiza en {cur_a} y «{sym_b}» en {cur_b}. En la v1 ambos activos "
                    "deben cotizar en la misma moneda: con monedas distintas el spread mezclaría "
                    "el movimiento de los precios con el del tipo de cambio. La conversión de "
                    "moneda queda como extensión futura de la herramienta."
                ),
            )
        )
        block_reasons.append(f"monedas distintas ({cur_a} y {cur_b})")
    currency_blocked = bool(unknown) or currency_mismatch

    # Alineación (siempre intersección; nunca se rellena)
    aligned = _align_clean(clean_a, clean_b)
    cal = aligned.calendar
    n_aligned = len(aligned.prices)
    n_train = int(math.floor(n_aligned * train_fraction))
    if aligned.only_in_a or aligned.only_in_b:
        detail = (
            " En un par cripto + bolsa esto incluye los fines de semana y los festivos de la "
            "bolsa, en los que la cripto sí cotiza."
            if cal == CalendarType.MIXED
            else ""
        )
        issues.append(
            ValidationIssue(
                code="FECHAS_NO_COMUNES",
                severity=Severity.INFO,
                details={"solo_en_a": len(aligned.only_in_a), "solo_en_b": len(aligned.only_in_b)},
                message=(
                    f"Se usan solo las fechas comunes a ambas series: se descartaron "
                    f"{format_int(len(aligned.only_in_a))} fecha(s) presentes solo en «{sym_a}» y "
                    f"{format_int(len(aligned.only_in_b))} presentes solo en «{sym_b}». No se "
                    f"rellenan.{detail}"
                ),
            )
        )

    # Precios ≤ 0 en las fechas que realmente se usarían
    non_positive = aligned.prices.index[(aligned.prices <= 0).any(axis=1).to_numpy()]
    log_spread_allowed = len(non_positive) == 0
    if not log_spread_allowed:
        np_dates = _to_dates(non_positive)
        if spread_type == SpreadType.LOG:
            issues.append(
                ValidationIssue(
                    code="SPREAD_LOG_BLOQUEADO",
                    severity=Severity.ERROR,
                    dates=np_dates,
                    message=(
                        f"El spread logarítmico no se puede calcular: hay precios menores o iguales "
                        f"a cero en {len(np_dates)} fecha(s) común(es) ({format_dates(np_dates)}). "
                        "El logaritmo exige precios estrictamente positivos. Revisa los datos o "
                        "usa el spread en precios."
                    ),
                )
            )
            block_reasons.append("hay precios ≤ 0 y el spread logarítmico exige precios positivos")
        else:
            issues.append(
                ValidationIssue(
                    code="PRECIOS_NO_POSITIVOS",
                    severity=Severity.WARNING,
                    dates=np_dates,
                    message=(
                        f"Hay precios menores o iguales a cero en {len(np_dates)} fecha(s) "
                        f"común(es) ({format_dates(np_dates)}). El spread en precios se puede "
                        "calcular, pero el logarítmico queda bloqueado. Comprueba que no sea un "
                        "error de datos."
                    ),
                )
            )

    # Historial mínimo en formación
    insufficient = n_train < min_train_obs
    if insufficient:
        needed = _min_total_obs(min_train_obs, train_fraction)
        years = needed / periods_per_year(cal)
        issues.append(
            ValidationIssue(
                code="DATOS_INSUFICIENTES",
                severity=Severity.ERROR,
                details={
                    "n_alineadas": n_aligned,
                    "n_formacion": n_train,
                    "minimo_formacion": min_train_obs,
                    "total_necesario": needed,
                },
                message=(
                    f"Datos insuficientes: hay {format_int(n_aligned)} observaciones comunes y la "
                    f"muestra de formación ({format_pct(train_fraction)} del total) tendría "
                    f"{format_int(n_train)}; se necesitan al menos {format_int(min_train_obs)} en "
                    f"formación, es decir, unas {format_int(needed)} observaciones comunes en "
                    f"total (≈ {format_decimal(years)} años de días de negociación). Amplía el "
                    "rango de fechas o elige activos con más historia en común."
                ),
            )
        )
        block_reasons.append(
            f"solo {format_int(n_train)} observaciones en formación (mínimo {format_int(min_train_obs)})"
        )

    if cal == CalendarType.MIXED:
        issues.append(
            ValidationIssue(
                code="DESFASE_CIERRE_CRIPTO",
                severity=Severity.WARNING,
                message=CLOSE_TIME_MISMATCH_WARNING,
            )
        )

    blocked = bool(block_reasons)
    if blocked:
        status = "Análisis bloqueado: " + "; ".join(block_reasons) + "."
    else:
        status = _usable_status(issues, n_aligned, n_train)

    report = ValidationReport(
        series=[val_a, val_b],
        calendar=cal,
        n_aligned=n_aligned,
        n_train_estimate=n_train,
        min_train_obs=min_train_obs,
        only_in_a=len(aligned.only_in_a),
        only_in_b=len(aligned.only_in_b),
        currency_mismatch=currency_mismatch,
        log_spread_allowed=log_spread_allowed,
        insufficient_data=insufficient,
        blocked=blocked,
        issues=issues,
        status_message=status,
    )
    return report, (None if currency_blocked else aligned)
