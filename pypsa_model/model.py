from __future__ import annotations
import pandas as pd
pd.options.future.infer_string = False  

import numpy as np
import pypsa

from .schemas import validate_inputs, validate_params

def _to_numpy_series(s: pd.Series) -> pd.Series:
    """Convierte ArrowStringArray / nullable dtypes a numpy float64."""
    return pd.Series(s.to_numpy(dtype=float, na_value=0.0), index=s.index)


def _clip_pu(s: pd.Series) -> pd.Series:
    return s.clip(lower=0.0, upper=1.0)


def _safe_series(name: str, s: pd.Series) -> pd.Series:
    if not isinstance(s, pd.Series):
        raise TypeError(f"{name} must be a pandas.Series")
    return s


def _extract_marginal_prices(network: pypsa.Network) -> pd.Series | None:
    """
    Best-effort extraction of marginal prices (shadow prices).
    Availability depends on PyPSA/linopy versions.
    """
    # Many PyPSA versions provide: network.buses_t.marginal_price (DataFrame)
    try:
        mp = network.buses_t.marginal_price
        if isinstance(mp, pd.DataFrame) and mp.shape[1] == 1:
            return mp.iloc[:, 0]
        if isinstance(mp, pd.Series):
            return mp
        # If multiple buses, caller can select later
        return None
    except Exception:
        return None


def run_dispatch(inputs: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Public API: run optimization for each isolated system and return results.

    Returns a dict:
      {
        "systems": {sys: {... per-system results ...}},
        "metadata": {"warnings": [...], "errors": [...], "solver": "..."}
      }
    """
    meta_warnings = []
    meta_errors = []

    vin, _ = validate_inputs(inputs)
    vpa, _ = validate_params(params)

    meta_warnings += vin.warnings + vpa.warnings
    if not vin.ok:
        meta_errors += vin.errors
    if not vpa.ok:
        meta_errors += vpa.errors

    if meta_errors:
        return {"systems": {}, "metadata": {"ok": False, "errors": meta_errors, "warnings": meta_warnings}}

    time_index: pd.DatetimeIndex = inputs["time_index"]
    systems = inputs["systems"]

    mc = params["marginal_cost_USD_per_MWh"]
    VOLL = float(params["VOLL_USD_per_MWh"])
    eff_store = float(params["battery"]["eff_store"])
    eff_dispatch = float(params["battery"]["eff_dispatch"])

    results_all: Dict[str, Any] = {}

    
    for sys in systems:
        n = pypsa.Network()
        n.set_snapshots(time_index)

        # Single bus per system
        bus = sys
        n.add("Bus", bus, carrier="AC")

        # Load
        demand = _to_numpy_series(_safe_series(f"demand_MW[{sys}]", inputs["demand_MW"][sys]))
        n.add("Load", f"load_{sys}", bus=bus, p_set=demand.values)

        cap = inputs["capacity_MW"][sys]
        p_nom_thermal = float(cap["thermal"])
        p_nom_solar = float(cap["solar"])
        p_nom_wind = float(cap["wind"])
        bat_power = float(cap["battery_power"])
        bat_energy = float(cap["battery_energy"])
        p_nom_hydro = float(cap.get("hydro", 0))
        hydro_budget_MWh = float(cap.get("hydro_energy_budget_MWh", 0))


        # Thermal generator
        if p_nom_thermal > 0:
            n.add(
                "Generator",
                f"thermal_{sys}",
                bus=bus,
                p_nom=p_nom_thermal,
                marginal_cost=float(mc["thermal"]),
                p_min_pu=0.35,
                p_max_pu=1.0,
            )
        
        # Hydro con energy budget

        if p_nom_hydro > 0:
            # Derivar p_max_pu como fracción del budget disponible por hora
            # Budget se distribuye uniformemente como techo de disponibilidad
            n_hours = len(time_index)
            max_dispatch_per_hour = min(p_nom_hydro, hydro_budget_MWh / n_hours)
            hydro_pu_val = max_dispatch_per_hour / p_nom_hydro  # escalar 0-1

            n.add(
                "StorageUnit",
                f"hydro_{sys}",
                bus=bus,
                p_nom=p_nom_hydro,
                max_hours=hydro_budget_MWh / p_nom_hydro,  # horas de descarga completa
                efficiency_store=0.0,        # embalse: no se recarga en el horizonte
                efficiency_dispatch=1.0,
                state_of_charge_initial=hydro_budget_MWh,
                cyclic_state_of_charge=False,  # budget no se renueva
                marginal_cost=float(mc.get("hydro", 5.0)),
                p_min_pu=0.0,
            )

        # VRE: solar & wind with time-varying availability p_max_pu
        vre = inputs["vre_pmaxpu"][sys]
        solar_pu = _clip_pu(_to_numpy_series(vre["solar"]))
        wind_pu  = _clip_pu(_to_numpy_series(vre["wind"]))

        if p_nom_solar > 0:
            n.add(
                "Generator",
                f"solar_{sys}",
                bus=bus,
                p_nom=p_nom_solar,
                marginal_cost=float(mc["solar"]),
                p_max_pu=solar_pu.values,
                p_min_pu=0.0,
            )

        if p_nom_wind > 0:
            n.add(
                "Generator",
                f"wind_{sys}",
                bus=bus,
                p_nom=p_nom_wind,
                marginal_cost=float(mc["wind"]),
                p_max_pu=wind_pu.values,
                p_min_pu=0.0,
            )

        # Load shedding as a very expensive generator
        # (represents unmet demand; optimization uses it only if necessary)
        n.add(
            "Generator",
            f"shedding_{sys}",
            bus=bus,
            p_nom=1e9,
            marginal_cost=VOLL,
            p_min_pu=0.0,
            p_max_pu=1.0,
        )
        

        # Battery (StorageUnit)
        if bat_power > 0 and bat_energy > 0:
            max_hours = bat_energy / max(bat_power, 1e-9)  # hours of full discharge at nominal power
            n.add(
                "StorageUnit",
                f"battery_{sys}",
                bus=bus,
                p_nom=bat_power,
                max_hours=max_hours,
                efficiency_store=eff_store,
                efficiency_dispatch=eff_dispatch,
                state_of_charge_initial=0.5 * bat_energy,
                cyclic_state_of_charge=True,
                marginal_cost=0.0,
            )

        # Solve
        try:
            n.optimize(solver_name="highs", include_objective_constant=False)
            ok = True
            solver_status = "ok"
        except Exception as e:
            ok = False
            solver_status = f"failed: {type(e).__name__}: {e}"

        if not ok:
            results_all[sys] = {
                "ok": False,
                "error": solver_status,
            }
            continue

        # Extract dispatch per tech
        gen_p = n.generators_t.p.copy() if hasattr(n, "generators_t") else pd.DataFrame(index=time_index)
        # Identify tech by name prefix
        tech_cols = {}
        for col in gen_p.columns:
            if col.startswith("thermal_"):
                tech = "thermal"
            elif col.startswith("solar_"):
                tech = "solar"
            elif col.startswith("wind_"):
                tech = "wind"
            elif col.startswith("shedding_"):
                tech = "shedding"
            else:
                tech = "other"
            tech_cols[col] = tech

        dispatch_by_tech = pd.DataFrame(index=time_index)
        for tech in ["thermal", "solar", "wind"]:
            cols = [c for c, t in tech_cols.items() if t == tech]
            dispatch_by_tech[tech] = gen_p[cols].sum(axis=1) if cols else 0.0

        shedding = gen_p[[c for c, t in tech_cols.items() if t == "shedding"]].sum(axis=1)
        shedding.name = "shedding_MW"

        # Curtailment for VRE
        available_solar = p_nom_solar * solar_pu
        used_solar = dispatch_by_tech["solar"]
        curtail_solar = (available_solar - used_solar).clip(lower=0.0)

        available_wind = p_nom_wind * wind_pu
        used_wind = dispatch_by_tech["wind"]
        curtail_wind = (available_wind - used_wind).clip(lower=0.0)

        curtailment = (curtail_solar + curtail_wind)
        curtailment.name = "curtailment_MW"

        # Battery SOC
        soc = None
        if len(n.storage_units.index) > 0:
            try:
                soc_df = n.storage_units_t.state_of_charge
                if isinstance(soc_df, pd.DataFrame) and soc_df.shape[1] == 1:
                    soc = soc_df.iloc[:, 0].rename("soc_MWh")
            except Exception:
                soc = None

        # Marginal price (best effort)
        mp = _extract_marginal_prices(n)
        if mp is not None:
            mp = mp.rename("marginal_price_USD_per_MWh")

        # Total cost (best effort)
        total_cost = None
        try:
            # objective value in many versions:
            if hasattr(n, "objective") and n.objective is not None:
                total_cost = float(n.objective)
        except Exception:
            total_cost = None

        results_all[sys] = {
            "ok": True,
            "solver_status": solver_status,
            "dispatch_MW": dispatch_by_tech,
            "demand_MW": demand.rename("demand_MW"),
            "shedding_MW": shedding,
            "curtailment_MW": curtailment,
            "soc_MWh": soc,
            "marginal_price_USD_per_MWh": mp,
            "total_cost_USD": total_cost,
        }

    return {
        "systems": results_all,
        "metadata": {"ok": True, "errors": meta_errors, "warnings": meta_warnings},
    }
