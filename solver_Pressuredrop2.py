# steam_generator_counterflow_with_pressure_drop.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI

FLIP_PLOT_LEFT_RIGHT = True


# ============================================================
# Basic correlations
# ============================================================

def dittus_boelter(Re, Pr, k, Dh, heating=True):
    Re = max(Re, 1.0)
    Pr = max(Pr, 1e-6)
    n = 0.4 if heating else 0.3
    Nu = 0.023 * Re**0.8 * Pr**n
    return Nu * k / Dh


def shah_boiling_correlation(h_l, q_flux, G, h_fg, quality):
    x = np.clip(quality, 1e-6, 0.999999)
    q_flux = max(q_flux, 1.0)
    Bo = q_flux / (G * h_fg)

    enhancement = (
        1.0
        + 3000.0 * Bo**0.86
        + 1.12 * (x / (1.0 - x))**0.75
    )

    return h_l * enhancement


# ============================================================
# Friction factor and pressure drop correlations
# ============================================================

def friction_factor_darcy(Re):
    """
    Darcy friction factor.
    Laminar : f = 64 / Re
    Turbulent : Blasius correlation
    """
    Re = max(Re, 1.0)

    if Re < 2300:
        return 64.0 / Re
    else:
        return 0.3164 / Re**0.25


def single_phase_dpdz(G, rho, mu, Dh):
    """
    Single-phase frictional pressure gradient [Pa/m]
    dp/dz = f * G^2 / (2 rho Dh)
    """
    Re = G * Dh / mu
    f = friction_factor_darcy(Re)
    dpdz = f * G**2 / (2.0 * rho * Dh)

    return dpdz, Re, f


def separated_flow_pressure_drop(
    quality,
    G,
    Dh,
    rho_l,
    rho_g,
    mu_l,
    mu_g,
    model="Mishima-Hibiki",
    channel_shape="circular"
):
    """
    Two-phase separated-flow pressure drop model.

    (dP/dz)_TP = (dP/dz)_f * phi_f^2
    phi_f^2 = 1 + C / X + 1 / X^2
    X = sqrt((dP/dz)_f / (dP/dz)_g)

    Default model: Mishima & Hibiki (1996).
    """

    x = np.clip(quality, 1e-6, 0.999999)

    G_l = max(G * (1.0 - x), 1e-12)
    G_g = max(G * x, 1e-12)

    dpdz_l, Re_l, f_l = single_phase_dpdz(G_l, rho_l, mu_l, Dh)
    dpdz_g, Re_g, f_g = single_phase_dpdz(G_g, rho_g, mu_g, Dh)

    X = np.sqrt(max(dpdz_l, 1e-30) / max(dpdz_g, 1e-30))
    model_key = model.strip().lower()
    shape_key = channel_shape.strip().lower()

    if model_key not in ("mishima-hibiki", "mishima & hibiki", "mishima_hibiki"):
        raise ValueError("Unknown pressure drop model. Use 'Mishima-Hibiki'.")

    if shape_key == "rectangular":
        C = 21.0 * (1.0 - np.exp(-0.319 * Dh))
    elif shape_key == "circular":
        C = 21.0 * (1.0 - np.exp(-0.333 * Dh))
    else:
        raise ValueError("Unknown channel shape. Use 'circular' or 'rectangular'.")

    phi2 = 1.0 + C / X + 1.0 / X**2

    dpdz_tp = dpdz_l * phi2

    return dpdz_tp, phi2, X, Re_l, Re_g


# ============================================================
# Cold-side state
# ============================================================

def get_cold_state(Hc, P_cold, fluid_cold, H_f, H_g, H_fg, T_sat):

    if Hc < H_f:
        T = PropsSI("T", "P", P_cold, "H", Hc, fluid_cold)
        quality = 0.0
        phase = "subcooled"

        rho = PropsSI("D", "P", P_cold, "H", Hc, fluid_cold)
        mu = PropsSI("V", "P", P_cold, "H", Hc, fluid_cold)
        k = PropsSI("L", "P", P_cold, "H", Hc, fluid_cold)
        cp = PropsSI("C", "P", P_cold, "H", Hc, fluid_cold)

    elif H_f <= Hc <= H_g:
        T = T_sat
        quality = (Hc - H_f) / H_fg
        phase = "boiling"

        rho = PropsSI("D", "P", P_cold, "Q", 0, fluid_cold)
        mu = PropsSI("V", "P", P_cold, "Q", 0, fluid_cold)
        k = PropsSI("L", "P", P_cold, "Q", 0, fluid_cold)
        cp = PropsSI("C", "P", P_cold, "Q", 0, fluid_cold)

    else:
        T = PropsSI("T", "P", P_cold, "H", Hc, fluid_cold)
        quality = 1.0
        phase = "superheated_vapor"

        rho = PropsSI("D", "P", P_cold, "H", Hc, fluid_cold)
        mu = PropsSI("V", "P", P_cold, "H", Hc, fluid_cold)
        k = PropsSI("L", "P", P_cold, "H", Hc, fluid_cold)
        cp = PropsSI("C", "P", P_cold, "H", Hc, fluid_cold)

    return T, quality, phase, rho, mu, k, cp


def get_cold_heat_transfer_coefficient(
    phase,
    Re_c,
    Pr_c,
    k_c,
    D_inner,
    q_flux,
    G_cold,
    H_fg,
    quality_local
):

    if phase == "subcooled":

        h_cold = dittus_boelter(
            Re=Re_c,
            Pr=Pr_c,
            k=k_c,
            Dh=D_inner,
            heating=True
        )
        correlation = "Dittus-Boelter liquid"

    elif phase == "boiling":

        h_l = dittus_boelter(
            Re=Re_c,
            Pr=Pr_c,
            k=k_c,
            Dh=D_inner,
            heating=True
        )

        h_cold = shah_boiling_correlation(
            h_l=h_l,
            q_flux=q_flux,
            G=G_cold,
            h_fg=H_fg,
            quality=quality_local
        )

        correlation = "Shah boiling"

    elif phase == "superheated_vapor":

        h_cold = dittus_boelter(
            Re=Re_c,
            Pr=Pr_c,
            k=k_c,
            Dh=D_inner,
            heating=True
        )
        correlation = "Dittus-Boelter vapor"

    else:
        h_cold = dittus_boelter(
            Re=Re_c,
            Pr=Pr_c,
            k=k_c,
            Dh=D_inner,
            heating=True
        )
        correlation = "Dittus-Boelter"

    return h_cold, correlation


# ============================================================
# Geometry and input conditions
# ============================================================

L = 5.0
N = 100
dx = L / N

D_inner = 0.012
D_outer = 0.016

A_flow = np.pi * D_inner**2 / 4.0
P_heat = np.pi * D_inner

t_wall = (D_outer - D_inner) / 2.0
k_wall = 16.0

fluid_hot = "Water"
fluid_cold = "Water"

m_dot_hot = 0.40
m_dot_cold = 0.08

P_hot = 15e6
P_cold = 5e6

T_hot_in = 580.0 + 273.15
T_cold_in = 200.0 + 273.15

pressure_drop_model = "Mishima-Hibiki"
channel_shape = "circular"


# ============================================================
# Arrays
# ============================================================

x = np.linspace(0.0, L, N + 1)

T_hot = np.zeros(N + 1)
H_hot = np.zeros(N + 1)

T_cold = np.zeros(N + 1)
H_cold = np.zeros(N + 1)

quality = np.zeros(N + 1)
phase = [""] * (N + 1)

h_hot_arr = np.zeros(N)
h_cold_arr = np.zeros(N)
U_arr = np.zeros(N)
q_arr = np.zeros(N)
q_flux_arr = np.zeros(N)

dpdz_hot_arr = np.zeros(N)
dpdz_cold_arr = np.zeros(N)
dP_hot_cell = np.zeros(N)
dP_cold_cell = np.zeros(N)
dP_cold_cumulative = np.zeros(N + 1)
dP_cold_cumulative_cell = np.zeros(N)

phi2_arr = np.zeros(N)
X_arr = np.zeros(N)

cold_correlation = [""] * N
pressure_correlation = [""] * N


# ============================================================
# Initial conditions
# ============================================================

T_hot[0] = T_hot_in
H_hot[0] = PropsSI("H", "P", P_hot, "T", T_hot_in, fluid_hot)

T_cold[N] = T_cold_in
H_cold[N] = PropsSI("H", "P", P_cold, "T", T_cold_in, fluid_cold)


# ============================================================
# Saturation properties
# ============================================================

T_sat = PropsSI("T", "P", P_cold, "Q", 0, fluid_cold)
H_f = PropsSI("H", "P", P_cold, "Q", 0, fluid_cold)
H_g = PropsSI("H", "P", P_cold, "Q", 1, fluid_cold)
H_fg = H_g - H_f

rho_l_sat = PropsSI("D", "P", P_cold, "Q", 0, fluid_cold)
rho_g_sat = PropsSI("D", "P", P_cold, "Q", 1, fluid_cold)

mu_l_sat = PropsSI("V", "P", P_cold, "Q", 0, fluid_cold)
mu_g_sat = PropsSI("V", "P", P_cold, "Q", 1, fluid_cold)

G_hot = m_dot_hot / A_flow
G_cold = m_dot_cold / A_flow


# ============================================================
# Main calculation
# ============================================================

for i in range(N):

    ih = i
    ic = N - i

    # -----------------------------
    # Hot side
    # -----------------------------
    T_hot[ih] = PropsSI("T", "P", P_hot, "H", H_hot[ih], fluid_hot)

    rho_h = PropsSI("D", "P", P_hot, "H", H_hot[ih], fluid_hot)
    mu_h = PropsSI("V", "P", P_hot, "H", H_hot[ih], fluid_hot)
    k_h = PropsSI("L", "P", P_hot, "H", H_hot[ih], fluid_hot)
    cp_h = PropsSI("C", "P", P_hot, "H", H_hot[ih], fluid_hot)

    V_h = m_dot_hot / (rho_h * A_flow)
    Re_h = rho_h * V_h * D_inner / mu_h
    Pr_h = cp_h * mu_h / k_h

    h_hot = dittus_boelter(
        Re=Re_h,
        Pr=Pr_h,
        k=k_h,
        Dh=D_inner,
        heating=False
    )

    dpdz_hot, _, _ = single_phase_dpdz(
        G=G_hot,
        rho=rho_h,
        mu=mu_h,
        Dh=D_inner
    )

    # -----------------------------
    # Cold side
    # -----------------------------
    (
        T_cold[ic],
        quality[ic],
        phase[ic],
        rho_c,
        mu_c,
        k_c,
        cp_c
    ) = get_cold_state(
        H_cold[ic],
        P_cold,
        fluid_cold,
        H_f,
        H_g,
        H_fg,
        T_sat
    )

    V_c = m_dot_cold / (rho_c * A_flow)
    Re_c = rho_c * V_c * D_inner / mu_c
    Pr_c = cp_c * mu_c / k_c

    q_flux_guess = 50000.0

    h_cold, corr_name = get_cold_heat_transfer_coefficient(
        phase=phase[ic],
        Re_c=Re_c,
        Pr_c=Pr_c,
        k_c=k_c,
        D_inner=D_inner,
        q_flux=q_flux_guess,
        G_cold=G_cold,
        H_fg=H_fg,
        quality_local=quality[ic]
    )

    U = 1.0 / (
        1.0 / h_hot
        + t_wall / k_wall
        + 1.0 / h_cold
    )

    dT = T_hot[ih] - T_cold[ic]

    if dT <= 0.0:
        q = 0.0
        q_flux_local = 0.0

    else:
        q = U * P_heat * dx * dT
        q_flux_local = q / (P_heat * dx)

        if phase[ic] == "boiling":

            h_cold, corr_name = get_cold_heat_transfer_coefficient(
                phase=phase[ic],
                Re_c=Re_c,
                Pr_c=Pr_c,
                k_c=k_c,
                D_inner=D_inner,
                q_flux=q_flux_local,
                G_cold=G_cold,
                H_fg=H_fg,
                quality_local=quality[ic]
            )

            U = 1.0 / (
                1.0 / h_hot
                + t_wall / k_wall
                + 1.0 / h_cold
            )

            q = U * P_heat * dx * dT
            q_flux_local = q / (P_heat * dx)

    # -----------------------------
    # Cold-side pressure drop
    # -----------------------------
    if phase[ic] == "boiling":

        dpdz_cold, phi2, X, Re_l, Re_g = separated_flow_pressure_drop(
            quality=quality[ic],
            G=G_cold,
            Dh=D_inner,
            rho_l=rho_l_sat,
            rho_g=rho_g_sat,
            mu_l=mu_l_sat,
            mu_g=mu_g_sat,
            model=pressure_drop_model,
            channel_shape=channel_shape
        )

        pressure_corr = pressure_drop_model

    else:
        dpdz_cold, _, _ = single_phase_dpdz(
            G=G_cold,
            rho=rho_c,
            mu=mu_c,
            Dh=D_inner
        )

        phi2 = 1.0
        X = np.nan
        pressure_corr = "Darcy-Weisbach single-phase"

    # -----------------------------
    # Save cell results
    # -----------------------------
    h_hot_arr[i] = h_hot
    h_cold_arr[i] = h_cold
    U_arr[i] = U
    q_arr[i] = q
    q_flux_arr[i] = q_flux_local

    dpdz_hot_arr[i] = dpdz_hot
    dpdz_cold_arr[i] = dpdz_cold

    dP_hot_cell[i] = dpdz_hot * dx
    dP_cold_cell[i] = dpdz_cold * dx

    phi2_arr[i] = phi2
    X_arr[i] = X

    cold_correlation[i] = corr_name
    pressure_correlation[i] = pressure_corr

    # -----------------------------
    # Update enthalpy
    # -----------------------------
    H_hot[ih + 1] = H_hot[ih] - q / m_dot_hot
    H_cold[ic - 1] = H_cold[ic] + q / m_dot_cold


# ============================================================
# Final node state conversion
# ============================================================

for i in range(N + 1):

    T_hot[i] = PropsSI("T", "P", P_hot, "H", H_hot[i], fluid_hot)

    (
        T_cold[i],
        quality[i],
        phase[i],
        rho_c,
        mu_c,
        k_c,
        cp_c
    ) = get_cold_state(
        H_cold[i],
        P_cold,
        fluid_cold,
        H_f,
        H_g,
        H_fg,
        T_sat
    )


# ============================================================
# Cumulative pressure drop
# Cold water flows from x = L to x = 0
# Therefore cumulative dP is accumulated from right to left.
# ============================================================

dP_cold_cumulative[N] = 0.0

for i in range(N - 1, -1, -1):
    dP_cold_cumulative[i] = dP_cold_cumulative[i + 1] + dP_cold_cell[i]

dP_cold_cumulative_cell[:] = dP_cold_cumulative[:-1]


# ============================================================
# Dataframes
# ============================================================

df_node = pd.DataFrame({
    "x_m": x,
    "T_hot_C": T_hot - 273.15,
    "T_cold_C": T_cold - 273.15,
    "cold_quality": quality,
    "cold_phase": phase,
    "cold_cumulative_dP_Pa": dP_cold_cumulative,
    "cold_cumulative_dP_kPa": dP_cold_cumulative / 1000.0,
})

df_cell = pd.DataFrame({
    "cell": np.arange(N),
    "x_mid_m": 0.5 * (x[:-1] + x[1:]),
    "h_hot_W_m2K": h_hot_arr,
    "h_cold_W_m2K": h_cold_arr,
    "U_W_m2K": U_arr,
    "q_W": q_arr,
    "q_flux_W_m2": q_flux_arr,
    "dpdz_hot_Pa_m": dpdz_hot_arr,
    "dpdz_cold_Pa_m": dpdz_cold_arr,
    "dP_hot_cell_Pa": dP_hot_cell,
    "dP_cold_cell_Pa": dP_cold_cell,
    "cold_cumulative_dP_Pa": dP_cold_cumulative_cell,
    "cold_cumulative_dP_kPa": dP_cold_cumulative_cell / 1000.0,
    "phi2": phi2_arr,
    "X": X_arr,
    "cold_heat_transfer_correlation": cold_correlation,
    "pressure_drop_correlation": pressure_correlation
})

df_node.to_csv(
    "steam_generator_node_results_with_pressure_drop.csv",
    index=False,
    encoding="utf-8-sig"
)

df_cell.to_csv(
    "steam_generator_cell_results_with_pressure_drop.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# Plot : temperature + pressure drop
# ============================================================

fig, ax1 = plt.subplots(figsize=(12, 6))

ax1.plot(
    x,
    T_hot - 273.15,
    linewidth=3,
    marker="o",
    markersize=3,
    label="Hot water temperature"
)

ax1.plot(
    x,
    T_cold - 273.15,
    linewidth=3,
    marker="s",
    markersize=3,
    label="Cold water / steam temperature"
)

ax1.axhline(
    T_sat - 273.15,
    linestyle="--",
    linewidth=1.5,
    label=f"Cold saturation temperature = {T_sat - 273.15:.1f} °C"
)

boiling_indices = np.where(np.array(phase) == "boiling")[0]

if boiling_indices.size > 0:
    ax1.axvspan(
        x[boiling_indices[0]],
        x[boiling_indices[-1]],
        alpha=0.18,
        label="Boiling region"
    )

if FLIP_PLOT_LEFT_RIGHT:
    ax1.invert_xaxis()
    xlabel = "Position x [m]  (left = L, right = 0)"
else:
    xlabel = "Position x [m]  (left = 0, right = L)"

ax1.set_xlabel(xlabel)
ax1.set_ylabel("Temperature [°C]")
ax1.grid(True, linestyle="--", alpha=0.5)

ax2 = ax1.twinx()

ax2.plot(
    x,
    dP_cold_cumulative / 1000.0,
    linewidth=3,
    linestyle="-.",
    label="Cold-side cumulative pressure drop"
)

ax2.set_ylabel("Cold-side cumulative pressure drop [kPa]")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(
    lines1 + lines2,
    labels1 + labels2,
    loc="best"
)

plt.title(
    "Counter-flow Steam Generator: Temperature and Pressure Drop",
    fontsize=15,
    fontweight="bold"
)

plt.tight_layout()
plt.savefig(
    "steam_generator_temperature_pressure_drop.png",
    dpi=300
)
plt.close(fig)


# ============================================================
# Print results
# ============================================================

print("\n==============================")
print("Steam Generator Results")
print("==============================")

print("[Direction]")
print("Hot water  : left  -> right")
print("Cold water : right -> left")

print("\n[Temperature]")
print(f"Hot inlet  at x = 0 : {T_hot[0] - 273.15:.2f} °C")
print(f"Hot outlet at x = L : {T_hot[-1] - 273.15:.2f} °C")
print(f"Cold inlet  at x = L : {T_cold[-1] - 273.15:.2f} °C")
print(f"Cold outlet at x = 0 : {T_cold[0] - 273.15:.2f} °C")

print("\n[Cold-side phase change]")
print(f"T_sat at P_cold = {P_cold / 1e6:.2f} MPa : {T_sat - 273.15:.2f} °C")
print(f"Latent heat H_fg : {H_fg / 1e3:.2f} kJ/kg")
print(f"Cold outlet quality at x = 0 : {quality[0]:.3f}")
print(f"Cold outlet phase : {phase[0]}")

print("\n[Pressure drop]")
print(f"Pressure drop model in boiling region : {pressure_drop_model}")
print(f"Total cold-side pressure drop : {dP_cold_cumulative[0] / 1000.0:.3f} kPa")
print(f"Total hot-side pressure drop  : {np.sum(dP_hot_cell) / 1000.0:.3f} kPa")

print("\n[Cold-side heat transfer correlation]")
print(df_cell["cold_heat_transfer_correlation"].value_counts())

print("\n[Cold-side pressure drop correlation]")
print(df_cell["pressure_drop_correlation"].value_counts())

print("\nCSV saved:")
print("steam_generator_node_results_with_pressure_drop.csv")
print("steam_generator_cell_results_with_pressure_drop.csv")
print("Figure saved:")
print("steam_generator_temperature_pressure_drop.png")
print("==============================")
