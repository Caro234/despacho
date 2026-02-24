"""
vre_loader.py — Cargador de perfiles VRE para el Simulador de Despacho Económico.

FUENTE: Renewables.ninja (MERRA-2, 2019)
  Solar: Perfil_de_generación_solar_{SIN,BCA,BCS}.csv
  Eólica: Perfil_de_generación_eólica.xlsx (hojas SIN, BCA, BCS)

INTEGRACIÓN CON model.py:
  La función principal `load_vre_profiles()` devuelve el dict
  `vre_pmaxpu` que va directo al argumento `inputs["vre_pmaxpu"]`
  de `run_dispatch()`.

CAPACIDADES EMBEBIDAS (extraídas del header JSON de cada archivo):
  SIN  Solar:  7,881 MW   Wind: 7,573 MW
  BCA  Solar:    411 MW   Wind:    40 MW
  BCS  Solar:     98 MW   Wind:    50 MW

LÓGICA DE ALINEACIÓN TEMPORAL:
  Los perfiles son del año 2019. Se reutilizan como perfil representativo
  de cualquier año de simulación via `time_index` mapeando DOY+hora.
  Esto es estándar en modelos de despacho educativos (y en muchos
  modelos de investigación que usan un "año típico meteorológico").
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Capacidades instaladas embebidas (MW) — extraídas de los headers ninja
# Actualizar si se cambia el año base o la fuente
# ---------------------------------------------------------------------------
_NINJA_CAPACITY_KW: Dict[str, Dict[str, float]] = {
    "SIN": {"solar": 7_881_000.0, "wind": 7_573_000.0},
    "BCA": {"solar":   410_982.0, "wind":    40_000.0},
    "BCS": {"solar":    97_943.0, "wind":    50_000.0},
}


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _parse_solar_csv(path: Path, sistema: str) -> pd.Series:
    """
    Lee un CSV de Renewables.ninja solar (3 líneas de comentario, luego
    columnas: time, local_time, electricity).

    Devuelve una Series UTC-aware indexada por timestamp, valores = p_max_pu ∈ [0,1].
    """
    df = pd.read_csv(path, comment="#", parse_dates=["time"])
    df = df.set_index("time")
    df.index = pd.DatetimeIndex(df.index, tz="UTC")

    cap_kw = _NINJA_CAPACITY_KW[sistema]["solar"]
    p_max_pu = (df["electricity"] / cap_kw).clip(0.0, 1.0)
    p_max_pu.name = f"solar_pu_{sistema}"

    _validate_profile(p_max_pu, f"solar/{sistema}")
    return p_max_pu


def _parse_wind_xlsx(path: Path, sistema: str) -> pd.Series:
    """
    Lee la hoja `sistema` del Excel de eólica de Renewables.ninja.
    Estructura: 3 filas de metadatos (row 0-2), fila 3 = encabezado.
    Columnas relevantes: time (col 0), electricity (col 2).

    Devuelve una Series UTC-aware, valores = p_max_pu ∈ [0,1].
    """
    df = pd.read_excel(
        path,
        sheet_name=sistema,
        skiprows=3,     # saltar 3 filas de metadatos
        header=0,
        usecols=[0, 2], # time + electricity; col 3 = wind_speed (no se usa aquí)
    )
    df.columns = ["time", "electricity"]
    df = df.dropna(subset=["time"])
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time")

    cap_kw = _NINJA_CAPACITY_KW[sistema]["wind"]
    p_max_pu = (df["electricity"] / cap_kw).clip(0.0, 1.0)
    p_max_pu.name = f"wind_pu_{sistema}"

    _validate_profile(p_max_pu, f"wind/{sistema}")
    return p_max_pu


def _validate_profile(s: pd.Series, label: str) -> None:
    """Invariantes básicas del perfil."""
    if s.isna().any():
        n = s.isna().sum()
        warnings.warn(f"[VRE] {label}: {n} NaNs encontrados — se rellenarán con 0.")
    if (s < 0).any() or (s > 1).any():
        warnings.warn(f"[VRE] {label}: valores fuera de [0,1] detectados — se clippearán.")
    if len(s) not in (8760, 8784):  # año normal o bisiesto
        warnings.warn(f"[VRE] {label}: {len(s)} filas (esperaba 8760/8784).")


def _align_to_time_index(
    profile: pd.Series,
    time_index: pd.DatetimeIndex,
) -> pd.Series:
    """
    Alinea un perfil anual (año 2019) a un time_index arbitrario.

    Estrategia: mapeo por (mes, día_del_mes, hora_UTC).
    Si el time_index es de un año diferente a 2019, el perfil se reutiliza
    como "año meteorológico representativo" (práctica estándar).

    Para timestamps sin correspondencia exacta (e.g. 29-Feb en año bisiesto)
    se hace interpolación lineal hacia adelante.
    """
    # Asegurar que el time_index sea UTC-aware
    if time_index.tz is None:
        time_index = time_index.tz_localize("UTC")
    else:
        time_index = time_index.tz_convert("UTC")

    # El perfil de 2019 puede no tener tz — normalizamos
    if profile.index.tz is None:
        profile.index = profile.index.tz_localize("UTC")

    # Construir lookup por (mes, dia, hora)
    lookup = {
        (ts.month, ts.day, ts.hour): val
        for ts, val in profile.items()
    }

    aligned_values = []
    for ts in time_index:
        key = (ts.month, ts.day, ts.hour)
        val = lookup.get(key, float("nan"))
        aligned_values.append(val)

    aligned = pd.Series(aligned_values, index=time_index, name=profile.name)

    # Rellenar NaNs residuales (e.g. 29-Feb) por interpolación
    if aligned.isna().any():
        n_nan = aligned.isna().sum()
        aligned = aligned.interpolate(method="time").fillna(0.0)
        warnings.warn(
            f"[VRE] {profile.name}: {n_nan} timestamps sin match en 2019 — interpolados."
        )

    return aligned.clip(0.0, 1.0)


# ---------------------------------------------------------------------------
# API PÚBLICA
# ---------------------------------------------------------------------------

def load_vre_profiles(
    solar_sin_path: str | Path,
    solar_bca_path: str | Path,
    solar_bcs_path: str | Path,
    wind_xlsx_path: str | Path,
    time_index: pd.DatetimeIndex,
    systems: Optional[list] = None,
) -> Dict[str, Dict[str, pd.Series]]:
    """
    Carga y alinea los perfiles VRE de Renewables.ninja al time_index de la simulación.

    Parámetros
    ----------
    solar_sin_path, solar_bca_path, solar_bcs_path :
        Rutas a los CSVs de generación solar por sistema.
    wind_xlsx_path :
        Ruta al Excel de generación eólica (hojas SIN, BCA, BCS).
    time_index :
        DatetimeIndex de la simulación (puede ser cualquier año, resolución horaria).
        Si no tiene timezone, se asume UTC.
    systems :
        Lista de sistemas a cargar. Default: ["SIN", "BCA", "BCS"].

    Retorna
    -------
    dict con estructura:
        {
          "SIN": {"solar": Series(0-1), "wind": Series(0-1)},
          "BCA": {"solar": Series(0-1), "wind": Series(0-1)},
          "BCS": {"solar": Series(0-1), "wind": Series(0-1)},
        }

    Este dict se pasa directamente como `inputs["vre_pmaxpu"]` en `run_dispatch()`.

    Ejemplo de uso
    --------------
    >>> from vre_loader import load_vre_profiles
    >>> import pandas as pd
    >>>
    >>> time_index = pd.date_range("2024-01-01", periods=168, freq="h", tz="UTC")
    >>>
    >>> vre = load_vre_profiles(
    ...     solar_sin_path="Perfil_de_generación_solar_SIN.csv",
    ...     solar_bca_path="Perfil_de_generación_solar_BCA.csv",
    ...     solar_bcs_path="Perfil_de_generación_solar_BCS.csv",
    ...     wind_xlsx_path="Perfil_de_generación_eólica.xlsx",
    ...     time_index=time_index,
    ... )
    >>>
    >>> # Integración directa con model.py:
    >>> inputs = {
    ...     "time_index": time_index,
    ...     "systems": ["SIN", "BCA", "BCS"],
    ...     "demand_MW": demand_dict,
    ...     "capacity_MW": capacity_dict,
    ...     "vre_pmaxpu": vre,          # <-- aquí van los perfiles
    ... }
    >>> results = run_dispatch(inputs, params)
    """
    if systems is None:
        systems = ["SIN", "BCA", "BCS"]

    solar_paths = {
        "SIN": Path(solar_sin_path),
        "BCA": Path(solar_bca_path),
        "BCS": Path(solar_bcs_path),
    }
    wind_path = Path(wind_xlsx_path)

    # Cargar perfiles base (año 2019, 8760 filas)
    solar_raw: Dict[str, pd.Series] = {}
    wind_raw: Dict[str, pd.Series] = {}

    for sys in systems:
        solar_raw[sys] = _parse_solar_csv(solar_paths[sys], sys)
        wind_raw[sys] = _parse_wind_xlsx(wind_path, sys)

    # Alinear al time_index de la simulación
    vre_pmaxpu: Dict[str, Dict[str, pd.Series]] = {}
    for sys in systems:
        vre_pmaxpu[sys] = {
            "solar": _align_to_time_index(solar_raw[sys], time_index),
            "wind":  _align_to_time_index(wind_raw[sys],  time_index),
        }

    return vre_pmaxpu


def ninja_installed_capacity_mw(sistema: str, tech: str) -> float:
    """
    Retorna la capacidad instalada (MW) usada como base de normalización
    en los archivos de Renewables.ninja.

    Útil para poblar `capacity_MW` en inputs de `run_dispatch()` si no
    se tiene un CSV propio de capacidades (semana 3).

    >>> ninja_installed_capacity_mw("SIN", "solar")
    7881.0
    """
    return _NINJA_CAPACITY_KW[sistema][tech] / 1000.0
