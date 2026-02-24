import pandas as pd
from pypsa_model.model import run_dispatch

# En capacity_MW de cada sistema:


def main():
    t = pd.date_range("2026-01-01", periods=48, freq="h")

    # Synthetic demand
    demand_sin = pd.Series(2000.0, index=t)
    demand_bca = pd.Series(450.0, index=t)
    demand_bcs = pd.Series(280.0, index=t)


    # Simple VRE profiles
    solar = pd.Series([0.0]*8 + [0.2]*4 + [0.7]*6 + [0.3]*4 + [0.0]*26, index=t).iloc[:48]
    wind = pd.Series(0.4, index=t)

    inputs = {
        "time_index": t,
        "systems": ["SIN", "BCA", "BCS"],
        "demand_MW": {"SIN": demand_sin, "BCA": demand_bca, "BCS": demand_bcs},
        "capacity_MW": {
            "SIN": {
                "thermal": 3000, "solar": 1500, "wind": 800,
                "hydro": 1200,
                "hydro_energy_budget_MWh": 14400,
                "battery_power": 200, "battery_energy": 800
            },
            "BCA": {
                "thermal": 700, "solar": 500, "wind": 150,
                "hydro": 0,
                "hydro_energy_budget_MWh": 0,
                "battery_power": 60, "battery_energy": 240
            },
            "BCS": {
                "thermal": 500, "solar": 350, "wind": 80,
                "hydro": 0,
                "hydro_energy_budget_MWh": 0,
                "battery_power": 40, "battery_energy": 160
            },
        },
        "vre_pmaxpu": {
            "SIN": {"solar": solar, "wind": wind},
            "BCA": {"solar": solar, "wind": wind},
            "BCS": {"solar": solar, "wind": wind},
        },
    }


    params = {
        "marginal_cost_USD_per_MWh": {
            "thermal": 60.0,
            "solar": 0.0,
            "wind": 0.0,
            "hydro": 5.0,   # ← agregar
        },
        "VOLL_USD_per_MWh": 2000.0,
        "battery": {"eff_store": 0.95, "eff_dispatch": 0.95},
    }

    results = run_dispatch(inputs, params)

    print("Metadata:", results["metadata"])
    for sys, r in results["systems"].items():
        print(f"\n=== {sys} ===")
        if not r.get("ok"):
            print("FAILED:", r.get("error"))
            continue
        print("total_cost_USD:", r.get("total_cost_USD"))
        print("dispatch head:\n", r["dispatch_MW"].head())
        print("shedding sum (MWh-ish):", float(r["shedding_MW"].sum()))
        print("curtailment sum:", float(r["curtailment_MW"].sum()))
        if r.get("soc_MWh") is not None:
            print("soc head:\n", r["soc_MWh"].head())
        if r.get("marginal_price_USD_per_MWh") is not None:
            print("price head:\n", r["marginal_price_USD_per_MWh"].head())
        else:
            print("price: not available in this environment/version")

if __name__ == "__main__":
    main()
