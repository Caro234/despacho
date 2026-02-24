from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

import pandas as pd


@dataclass
class ValidationResult:
    ok: bool
    errors: List[str]
    warnings: List[str]


def _require_keys(d: dict, keys: List[str], where: str) -> List[str]:
    missing = [k for k in keys if k not in d]
    return [f"Missing key '{k}' in {where}" for k in missing]


def validate_inputs(inputs: Dict[str, Any]) -> Tuple[ValidationResult, Dict[str, Any]]:
    """
    Validates and lightly normalizes the inputs contract.

    Expected:
    inputs = {
      "time_index": DatetimeIndex,
      "systems": ["SIN","BCA","BCS"],
      "demand_MW": {sys: Series},
      "capacity_MW": {sys: {"thermal":..., "solar":..., "wind":..., "battery_power":..., "battery_energy":...}},
      "vre_pmaxpu": {sys: {"solar": Series(0-1), "wind": Series(0-1)}}
    }
    """
    errors: List[str] = []
    warnings: List[str] = []

    if not isinstance(inputs, dict):
        return ValidationResult(False, ["inputs must be a dict"], []), inputs

    errors += _require_keys(
        inputs,
        ["time_index", "systems", "demand_MW", "capacity_MW", "vre_pmaxpu"],
        "inputs",
    )

    if errors:
        return ValidationResult(False, errors, warnings), inputs

    time_index = inputs["time_index"]
    systems = inputs["systems"]

    if not isinstance(time_index, pd.DatetimeIndex):
        errors.append("inputs['time_index'] must be a pandas.DatetimeIndex")

    if not isinstance(systems, list) or not all(isinstance(s, str) for s in systems):
        errors.append("inputs['systems'] must be a list[str]")

    demand = inputs["demand_MW"]
    cap = inputs["capacity_MW"]
    vre = inputs["vre_pmaxpu"]

    for sys in systems:
        if sys not in demand:
            errors.append(f"demand_MW missing system '{sys}'")
            continue
        if sys not in cap:
            errors.append(f"capacity_MW missing system '{sys}'")
            continue
        if sys not in vre:
            errors.append(f"vre_pmaxpu missing system '{sys}'")
            continue

        # demand series checks
        dser = demand[sys]
        if not isinstance(dser, pd.Series):
            errors.append(f"demand_MW['{sys}'] must be a pandas.Series")
        else:
            if not dser.index.equals(time_index):
                errors.append(f"demand_MW['{sys}'] index must equal time_index")
            if dser.isna().any():
                errors.append(f"demand_MW['{sys}'] contains NaNs")
            if (dser < 0).any():
                warnings.append(f"demand_MW['{sys}'] has negative values; check data")

        # capacity checks
        cdict = cap[sys]
        if not isinstance(cdict, dict):
            errors.append(f"capacity_MW['{sys}'] must be a dict")
        else:
            for k in ["thermal", "solar", "wind", "battery_power", "battery_energy"]:
                if k not in cdict:
                    errors.append(f"capacity_MW['{sys}'] missing '{k}'")
                else:
                    try:
                        float(cdict[k])
                    except Exception:
                        errors.append(f"capacity_MW['{sys}']['{k}'] must be numeric")

        # VRE p_max_pu checks
        vdict = vre[sys]
        if not isinstance(vdict, dict):
            errors.append(f"vre_pmaxpu['{sys}'] must be a dict")
        else:
            for tech in ["solar", "wind"]:
                if tech not in vdict:
                    errors.append(f"vre_pmaxpu['{sys}'] missing '{tech}'")
                    continue
                s = vdict[tech]
                if not isinstance(s, pd.Series):
                    errors.append(f"vre_pmaxpu['{sys}']['{tech}'] must be a pandas.Series")
                else:
                    if not s.index.equals(time_index):
                        errors.append(f"vre_pmaxpu['{sys}']['{tech}'] index must equal time_index")
                    if s.isna().any():
                        errors.append(f"vre_pmaxpu['{sys}']['{tech}'] contains NaNs")
                    if ((s < 0) | (s > 1)).any():
                        warnings.append(f"vre_pmaxpu['{sys}']['{tech}'] outside [0,1]; will be clipped")

    return ValidationResult(len(errors) == 0, errors, warnings), inputs


def validate_params(params: Dict[str, Any]) -> Tuple[ValidationResult, Dict[str, Any]]:
    """
    Expected:
    params = {
      "marginal_cost_USD_per_MWh": {"thermal": 60, "solar": 0, "wind": 0},
      "VOLL_USD_per_MWh": 2000,
      "battery": {"eff_store": 0.95, "eff_dispatch": 0.95}
    }
    """
    errors: List[str] = []
    warnings: List[str] = []

    if not isinstance(params, dict):
        return ValidationResult(False, ["params must be a dict"], []), params

    errors += _require_keys(params, ["marginal_cost_USD_per_MWh", "VOLL_USD_per_MWh", "battery"], "params")

    if errors:
        return ValidationResult(False, errors, warnings), params

    mc = params["marginal_cost_USD_per_MWh"]
    if not isinstance(mc, dict):
        errors.append("params['marginal_cost_USD_per_MWh'] must be a dict")
    else:
        for k in ["thermal", "solar", "wind"]:
            if k not in mc:
                errors.append(f"marginal_cost_USD_per_MWh missing '{k}'")
            else:
                try:
                    float(mc[k])
                except Exception:
                    errors.append(f"marginal_cost_USD_per_MWh['{k}'] must be numeric")

    try:
        float(params["VOLL_USD_per_MWh"])
    except Exception:
        errors.append("params['VOLL_USD_per_MWh'] must be numeric")

    bat = params["battery"]
    if not isinstance(bat, dict):
        errors.append("params['battery'] must be a dict")
    else:
        for k in ["eff_store", "eff_dispatch"]:
            if k not in bat:
                errors.append(f"battery missing '{k}'")
            else:
                val = float(bat[k])
                if not (0 < val <= 1):
                    warnings.append(f"battery['{k}'] is {val}, expected (0,1]")

    return ValidationResult(len(errors) == 0, errors, warnings), params
