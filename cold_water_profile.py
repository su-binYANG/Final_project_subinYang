# main_cold_profile.py

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI

from correlation import (
    dittus_boelter_h,
    shah_correlation,
    hem_pressure_drop,
    churchill_friction_factor,
)


# =========================================================
# Input data
# =========================================================
params = {
    "fluid_hot": "Water",
    "fluid_cold": "Water",

    "L": 5.0,
    "N": 100,

    "D_inner": 0.012,
    "D_outer": 0.016,
    "t_wall": 0.002,
    "k_wall": 16.0,

    "m_dot_hot": 0.4,
    "m_dot_cold": 0.08,

    "P_hot": 15000000,
    "P_cold": 6000000,

    "T_hot_in": 580.0,
    "T_cold_in": 420.0,

    "q_flux": 50000,

    "flow": "counter"
}


# =========================================================
# Main calculation
# =========================================================
def calculate_cold_profile(params):
    fluid = params["fluid_cold"]

    L = params["L"]
    N = params["N"]
    D_inner = params["D_inner"]
    m_dot = params["m_dot_cold"]
    P = params["P_cold"]
    T_in_C = params["T_cold_in"]
    q_flux = params["q_flux"]

    T_in = T_in_C + 273.15

    dx = L / N
    x_pos = np.linspace(0.0, L, N + 1)

    A_flow = np.pi * D_inner**2 / 4.0
    P_heated = np.pi * D_inner
    G = m_dot / A_flow

    # Saturation properties
    T_sat = PropsSI("T", "P", P, "Q", 0, fluid)
    T_sat_C = T_sat - 273.15

    h_in = PropsSI("H", "P", P, "T", T_in, fluid)
    h_f = PropsSI("H", "P", P, "Q", 0, fluid)
    h_g = PropsSI("H", "P", P, "Q", 1, fluid)
    h_fg = h_g - h_f

    rho_l = PropsSI("D", "P", P, "Q", 0, fluid)
    rho_g = PropsSI("D", "P", P, "Q", 1, fluid)

    mu_l = PropsSI("V", "P", P, "Q", 0, fluid)
    mu_g = PropsSI("V", "P", P, "Q", 1, fluid)

    k_l = PropsSI("L", "P", P, "Q", 0, fluid)
    k_g = PropsSI("L", "P", P, "Q", 1, fluid)

    cp_l = PropsSI("C", "P", P, "Q", 0, fluid)
    cp_g = PropsSI("C", "P", P, "Q", 1, fluid)

    h_cold = np.zeros(N + 1)
    T_cold_C = np.zeros(N + 1)
    quality = np.zeros(N + 1)
    phase = []
    htc = []
    dpdz = []

    h_cold[0] = h_in

    for i in range(N + 1):

        if i > 0:
            q_cell = q_flux * P_heated * dx
            dh = q_cell / m_dot
            h_cold[i] = h_cold[i - 1] + dh

        h_now = h_cold[i]

        # =================================================
        # 1. Subcooled liquid
        # =================================================
        if h_now < h_f:
            T_now = PropsSI("T", "P", P, "H", h_now, fluid)
            T_cold_C[i] = T_now - 273.15
            quality[i] = 0.0
            phase.append("subcooled")

            h_result = dittus_boelter_h(
                G=G,
                Dh=D_inner,
                k=k_l,
                mu=mu_l,
                cp=cp_l,
                heating=True
            )

            htc.append(h_result["h"])

            Re = h_result["Re"]
            f = churchill_friction_factor(Re, roughness=0.0, Dh=D_inner)
            dpdz_now = f * G**2 / (2.0 * rho_l * D_inner)
            dpdz.append(dpdz_now)

        # =================================================
        # 2. Boiling region
        # =================================================
        elif h_now <= h_g:
            T_cold_C[i] = T_sat_C
            xq = (h_now - h_f) / h_fg
            xq = max(0.0, min(1.0, xq))
            quality[i] = xq
            phase.append("boiling")

            h_sp = dittus_boelter_h(
                G=G,
                Dh=D_inner,
                k=k_l,
                mu=mu_l,
                cp=cp_l,
                heating=True
            )["h"]

            h_result = shah_correlation(
                h_sp=h_sp,
                q_flux=q_flux,
                G=G,
                h_fg=h_fg,
                x=xq,
                rho_g=rho_g,
                rho_l=rho_l,
                Dh=D_inner
            )

            htc.append(h_result["h_tp"])

            dp_result = hem_pressure_drop(
                G=G,
                Dh=D_inner,
                x=xq,
                rho_l=rho_l,
                rho_v=rho_g,
                mu_l=mu_l,
                mu_v=mu_g,
                viscosity_model="McAdams",
                roughness=0.0
            )

            dpdz.append(dp_result["dpdz"])

        # =================================================
        # 3. Superheated vapor
        # =================================================
        else:
            T_now = PropsSI("T", "P", P, "H", h_now, fluid)
            T_cold_C[i] = T_now - 273.15
            quality[i] = 1.0
            phase.append("superheated")

            h_result = dittus_boelter_h(
                G=G,
                Dh=D_inner,
                k=k_g,
                mu=mu_g,
                cp=cp_g,
                heating=True
            )

            htc.append(h_result["h"])

            Re = h_result["Re"]
            f = churchill_friction_factor(Re, roughness=0.0, Dh=D_inner)
            dpdz_now = f * G**2 / (2.0 * rho_g * D_inner)
            dpdz.append(dpdz_now)

    df_nodes = pd.DataFrame({
        "position_m": x_pos,
        "T_cold_C": T_cold_C,
        "H_cold_J_kg": h_cold,
        "quality": quality,
        "phase": phase,
        "h_cold_W_m2K": htc,
        "dpdz_Pa_m": dpdz,
    })

    df_cells = df_nodes.iloc[:-1].copy()
    df_cells["dx_m"] = dx

    total_dp = np.trapezoid(dpdz, x_pos)

    summary = {
        "T_sat_C": T_sat_C,
        "T_out_C": T_cold_C[-1],
        "quality_out": quality[-1],
        "total_dp_Pa": total_dp,
        "G_kg_m2s": G,
    }

    return df_nodes, df_cells, summary


# =========================================================
# Plot
# =========================================================
def plot_cold_profile(df_nodes, summary, params):
    x = df_nodes["position_m"].values
    T = df_nodes["T_cold_C"].values
    phase = df_nodes["phase"].values
    quality = df_nodes["quality"].values

    L = params["L"]
    P = params["P_cold"]
    q_flux = params["q_flux"]
    m_dot = params["m_dot_cold"]
    D_inner = params["D_inner"]

    T_sat_C = summary["T_sat_C"]

    boil_idx = np.where(phase == "boiling")[0]

    if len(boil_idx) > 0:
        x_boil_start = x[boil_idx[0]]
        x_boil_end = x[boil_idx[-1]]
    else:
        x_boil_start = None
        x_boil_end = None

    def format_position(value):
        return f"{value:.2f} m" if value is not None else "N/A"

    fig = plt.figure(figsize=(13, 8))
    ax = fig.add_axes([0.08, 0.42, 0.84, 0.50])

    mask_sub = phase == "subcooled"
    mask_boil = phase == "boiling"
    mask_sup = phase == "superheated"

    ax.plot(x[mask_sub], T[mask_sub], "b-", lw=2, drawstyle="steps-post")
    ax.plot(x[mask_boil], T[mask_boil], "r-", lw=2, drawstyle="steps-post")
    ax.plot(x[mask_sup], T[mask_sup], "g-", lw=2, drawstyle="steps-post")

    ax.plot(x, T, "bo", ms=3)

    ax.plot(x[0], T[0], "bo", ms=12, mfc="white", mew=1.5)
    ax.plot(x[-1], T[-1], "bo", ms=12, mfc="white", mew=1.5)

    if x_boil_start is not None:
        ax.plot(x_boil_start, T_sat_C, "ro", ms=12, mfc="white", mew=1.5)
        ax.vlines(x_boil_start, 300, T_sat_C, colors="k", linestyles="--", lw=1)

    if x_boil_end is not None:
        ax.plot(x_boil_end, T_sat_C, "go", ms=12, mfc="white", mew=1.5)
        ax.vlines(x_boil_end, 300, T_sat_C, colors="k", linestyles="--", lw=1)

    ax.text(0.23, 600, "① Subcooled\nLiquid Region", color="blue", fontsize=12)
    ax.text(1.82, 600, "② Boiling Region\n(Saturated at $T_{sat}$)", color="red", fontsize=12)
    ax.text(3.20, 600, "③ Superheated\nVapor Region", color="green", fontsize=12)

    ax.text(1.9, T_sat_C + 12, f"$T_{{sat}}$ = {T_sat_C:.1f} °C", fontsize=11)

    if x_boil_start is not None:
        ax.annotate("", xy=(0.0, 567), xytext=(x_boil_start, 567),
                    arrowprops=dict(arrowstyle="<->", color="black"))

    if x_boil_start is not None and x_boil_end is not None:
        ax.annotate("", xy=(x_boil_start, 567), xytext=(x_boil_end, 567),
                    arrowprops=dict(arrowstyle="<->", color="black"))

        ax.annotate("", xy=(x_boil_end, 567), xytext=(4.0, 567),
                    arrowprops=dict(arrowstyle="<->", color="black"))

        ax.text(x_boil_start - 0.30, 310,
                f"Onset of Boiling\nx = {x_boil_start:.2f} m",
                color="red", fontsize=11)

        ax.text(x_boil_end - 0.25, 310,
                f"End of Boiling\nx = {x_boil_end:.2f} m",
                color="green", fontsize=11)

    ax.text(-0.1, T[0] - 35,
            f"$T_{{in}}$ = {T[0]:.1f} °C",
            color="blue", fontsize=11)

    ax.text(L - 0.25, T[-1] - 35,
            f"$T_{{out}}$ = {T[-1]:.1f} °C",
            color="blue", fontsize=11)

    info = (
        f"$P$ = {P / 1e6:.2f} MPa\n"
        f"$q''$ = {q_flux:,.0f} W/m$^2$\n"
        f"$\\dot{{m}}$ = {m_dot:.2f} kg/s\n"
        f"$D_{{inner}}$ = {D_inner:.3f} m\n"
        f"$L$ = {L:.1f} m"
    )

    ax.text(4.45, 360, info, fontsize=11,
            bbox=dict(boxstyle="round", fc="white", ec="black"))

    ax.set_title("Cold Water / Steam Temperature Profile",
                 fontsize=16, fontweight="bold")
    ax.set_xlabel("Position [m]", fontsize=12)
    ax.set_ylabel("Temperature [°C]", fontsize=12)
    ax.set_xlim(-0.25, L + 0.35)
    ax.set_ylim(300, 700)
    ax.grid(True, linestyle="--", alpha=0.4)

    # =====================================================
    # Bottom results
    # =====================================================
    fig.text(0.03, 0.33, "Key Results", fontsize=12, fontweight="bold")

    key_text = (
        f"• Saturation Temperature  $T_{{sat}}$   :  {T_sat_C:.1f} °C\n\n"
        f"• Onset of Boiling (x)              :  {format_position(x_boil_start)}\n\n"
        f"• End of Boiling (x)                :  {format_position(x_boil_end)}\n\n"
        f"• Outlet Temperature                :  {T[-1]:.1f} °C "
        f"({phase[-1]})\n\n"
        f"• Outlet Quality                    :  {quality[-1]:.2f}\n\n"
        f"• Total Pressure Drop               :  {summary['total_dp_Pa']:.1f} Pa\n\n"
        f"• Total Length                      :  {L:.2f} m"
    )

    fig.text(0.035, 0.07, key_text, fontsize=10)

    fig.text(0.34, 0.33, "Node Results (first 5 rows)",
             fontsize=12, fontweight="bold")

    table_df = df_nodes.head(5).copy()
    table_df = table_df[[
        "position_m",
        "T_cold_C",
        "H_cold_J_kg",
        "quality",
        "phase"
    ]]

    table_df["position_m"] = table_df["position_m"].map(lambda v: f"{v:.3f}")
    table_df["T_cold_C"] = table_df["T_cold_C"].map(lambda v: f"{v:.2f}")
    table_df["H_cold_J_kg"] = table_df["H_cold_J_kg"].map(lambda v: f"{v:.0f}")
    table_df["quality"] = table_df["quality"].map(lambda v: f"{v:.3f}")

    ax_table = fig.add_axes([0.34, 0.13, 0.36, 0.17])
    ax_table.axis("off")

    table = ax_table.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        loc="center",
        cellLoc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 1.45)

    fig.text(0.77, 0.33, "Legend (Phase)", fontsize=12, fontweight="bold")

    fig.text(0.77, 0.29, "━", color="blue", fontsize=16)
    fig.text(0.80, 0.292, "subcooled liquid  ($T < T_{sat}$)", fontsize=10)

    fig.text(0.77, 0.26, "━", color="red", fontsize=16)
    fig.text(0.80, 0.262, "boiling region  ($T = T_{sat}$)", fontsize=10)

    fig.text(0.77, 0.23, "━", color="green", fontsize=16)
    fig.text(0.80, 0.232, "superheated vapor  ($T > T_{sat}$)", fontsize=10)

    fig.text(0.26, 0.04,
             "CSV saved:  cold_nodes.csv,  cold_cells.csv",
             color="blue", fontsize=10)

    fig.text(0.54, 0.04,
             "Figure saved:  cold_water_temperature.png",
             color="blue", fontsize=10)

    plt.savefig("cold_water_temperature.png", dpi=300, bbox_inches="tight")
    plt.show()


# =========================================================
# Run
# =========================================================
if __name__ == "__main__":
    df_nodes, df_cells, summary = calculate_cold_profile(params)

    df_nodes.to_csv("cold_nodes.csv", index=False, encoding="utf-8-sig")
    df_cells.to_csv("cold_cells.csv", index=False, encoding="utf-8-sig")

    print("===== Cold Water / Steam Profile Result =====")
    print(f"T_sat = {summary['T_sat_C']:.2f} °C")
    print(f"T_out = {summary['T_out_C']:.2f} °C")
    print(f"Outlet quality = {summary['quality_out']:.3f}")
    print(f"Total pressure drop = {summary['total_dp_Pa']:.2f} Pa")
    print("CSV saved: cold_nodes.csv, cold_cells.csv")
    print("Figure saved: cold_water_temperature.png")

    plot_cold_profile(df_nodes, summary, params)
