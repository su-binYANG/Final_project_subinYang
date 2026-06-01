# steam_generator_counterflow.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI

FLIP_PLOT_LEFT_RIGHT = True


# ============================================================
# Correlations
# ============================================================

def dittus_boelter(Re, Pr, k, Dh, heating=True):
    """
    Single-phase forced convection correlation.
    Heating fluid: n = 0.4
    Cooling fluid: n = 0.3
    """

    Re = max(Re, 1.0)
    Pr = max(Pr, 1e-6)

    n = 0.4 if heating else 0.3

    Nu = 0.023 * Re**0.8 * Pr**n

    h = Nu * k / Dh

    return h


def shah_boiling_correlation(h_l, q_flux, G, h_fg, quality):
    """
    Simplified Shah-type boiling heat transfer correlation.

    h_l     : liquid-only heat transfer coefficient [W/m2K]
    q_flux  : local heat flux [W/m2]
    G       : mass flux [kg/m2s]
    h_fg    : latent heat [J/kg]
    quality : vapor quality [-]
    """

    x = np.clip(quality, 1e-6, 0.999999)

    q_flux = max(q_flux, 1.0)

    Bo = q_flux / (G * h_fg)

    enhancement = (
        1.0
        + 3000.0 * Bo**0.86
        + 1.12 * (x / (1.0 - x))**0.75
    )

    h_tp = h_l * enhancement

    return h_tp


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
    """
    Cold-side heat transfer coefficient by phase.

    subcooled liquid  : Dittus-Boelter
    boiling region    : Shah-type boiling correlation
    superheated vapor : Dittus-Boelter
    """

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
# Geometry
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


# ============================================================
# Hot side : LEFT -> RIGHT
# ============================================================

fluid_hot = "Water"

m_dot_hot = 0.40
P_hot = 15e6
T_hot_in = 580.0 + 273.15


# ============================================================
# Cold side : RIGHT -> LEFT
# ============================================================

fluid_cold = "Water"

m_dot_cold = 0.08
P_cold = 5e6

T_cold_in_C = 200.0
T_cold_in = T_cold_in_C + 273.15


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
cold_correlation = [""] * N


# ============================================================
# Initial condition
# ============================================================

# Hot inlet: x = 0, left side
T_hot[0] = T_hot_in
H_hot[0] = PropsSI(
    "H",
    "P", P_hot,
    "T", T_hot_in,
    fluid_hot
)

# Cold inlet: x = L, right side
T_cold[N] = T_cold_in
H_cold[N] = PropsSI(
    "H",
    "P", P_cold,
    "T", T_cold_in,
    fluid_cold
)


# ============================================================
# Cold-side saturation properties
# ============================================================

T_sat = PropsSI("T", "P", P_cold, "Q", 0, fluid_cold)

H_f = PropsSI("H", "P", P_cold, "Q", 0, fluid_cold)
H_g = PropsSI("H", "P", P_cold, "Q", 1, fluid_cold)

H_fg = H_g - H_f

if T_cold_in >= T_sat:
    print(
        "[Warning] Cold inlet is not subcooled liquid at this pressure. "
        f"At P_cold = {P_cold / 1e6:.2f} MPa, "
        f"T_sat = {T_sat - 273.15:.2f} °C."
    )


# ============================================================
# Main calculation
# ============================================================

G_cold = m_dot_cold / A_flow

for i in range(N):

    # Hot side: left -> right
    ih = i

    # Cold side: right -> left
    ic = N - i

    # =====================================================
    # Hot side properties
    # =====================================================

    T_hot[ih] = PropsSI(
        "T",
        "P", P_hot,
        "H", H_hot[ih],
        fluid_hot
    )

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

    # =====================================================
    # Cold side state
    # =====================================================

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

    # =====================================================
    # First estimate of cold-side h
    # =====================================================

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

    # =====================================================
    # Overall heat transfer coefficient
    # =====================================================

    U = 1.0 / (
        1.0 / h_hot
        + t_wall / k_wall
        + 1.0 / h_cold
    )

    # =====================================================
    # Heat transfer
    # =====================================================

    dT = T_hot[ih] - T_cold[ic]

    if dT <= 0.0:

        q = 0.0
        q_flux_local = 0.0

    else:

        q = U * P_heat * dx * dT

        q_flux_local = q / (P_heat * dx)

        # If boiling, recalculate h using local heat flux
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

    # =====================================================
    # Save cell results
    # =====================================================

    h_hot_arr[i] = h_hot
    h_cold_arr[i] = h_cold
    U_arr[i] = U
    q_arr[i] = q
    q_flux_arr[i] = q_flux_local
    cold_correlation[i] = corr_name

    # =====================================================
    # Update hot side
    # Hot water loses heat
    # x = 0 -> L
    # =====================================================

    H_hot[ih + 1] = H_hot[ih] - q / m_dot_hot

    # =====================================================
    # Update cold side
    # Cold water gains heat
    # x = L -> 0
    # =====================================================

    H_cold[ic - 1] = H_cold[ic] + q / m_dot_cold


# ============================================================
# Final state conversion
# ============================================================

for i in range(N + 1):

    T_hot[i] = PropsSI(
        "T",
        "P", P_hot,
        "H", H_hot[i],
        fluid_hot
    )

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
# Dataframe
# ============================================================

df_node = pd.DataFrame({
    "x_m": x,
    "T_hot_C": T_hot - 273.15,
    "T_cold_C": T_cold - 273.15,
    "cold_superheat_C": np.maximum(T_cold - T_sat, 0.0),
    "H_hot_Jkg": H_hot,
    "H_cold_Jkg": H_cold,
    "quality": quality,
    "cold_phase": phase
})

df_cell = pd.DataFrame({
    "cell": np.arange(N),
    "x_mid_m": 0.5 * (x[:-1] + x[1:]),
    "h_hot_W_m2K": h_hot_arr,
    "h_cold_W_m2K": h_cold_arr,
    "U_W_m2K": U_arr,
    "q_W": q_arr,
    "q_flux_W_m2": q_flux_arr,
    "cold_correlation": cold_correlation
})

df_node.to_csv(
    "steam_generator_node_results.csv",
    index=False,
    encoding="utf-8-sig"
)

df_cell.to_csv(
    "steam_generator_cell_results.csv",
    index=False,
    encoding="utf-8-sig"
)

print(df_node)
print(df_cell)


# ============================================================
# Plot
# ============================================================

plt.figure(figsize=(11, 6))

plt.plot(
    x,
    T_hot - 273.15,
    color="red",
    linewidth=3,
    marker="o",
    markersize=3,
    label="Hot water: left → right"
)

plt.plot(
    x,
    T_cold - 273.15,
    color="blue",
    linewidth=3,
    marker="o",
    markersize=3,
    label="Cold water / steam: right → left"
)

plt.axhline(
    T_sat - 273.15,
    color="black",
    linestyle="--",
    linewidth=1.5,
    label=f"Cold-side saturation temperature = {T_sat - 273.15:.1f} °C"
)

boiling_indices = np.where(np.array(phase) == "boiling")[0]

if boiling_indices.size > 0:
    plt.axvspan(
        x[boiling_indices[0]],
        x[boiling_indices[-1]],
        color="skyblue",
        alpha=0.18,
        label="Boiling region: Shah correlation"
    )

superheated_indices = np.where(np.array(phase) == "superheated_vapor")[0]

if superheated_indices.size > 0:
    plt.axvspan(
        x[superheated_indices[0]],
        x[superheated_indices[-1]],
        color="orange",
        alpha=0.14,
        label="Superheated vapor region"
    )

plt.text(
    0.05 * L,
    T_hot[0] - 273.15 + 5,
    "Hot inlet",
    color="red"
)

plt.text(
    0.78 * L,
    T_hot[-1] - 273.15 + 5,
    "Hot outlet",
    color="red"
)

plt.text(
    0.75 * L,
    T_cold[-1] - 273.15 - 25,
    "Cold inlet",
    color="blue"
)

plt.text(
    0.03 * L,
    T_cold[0] - 273.15 + 5,
    "Cold outlet",
    color="blue"
)

plt.annotate(
    "Hot flow",
    xy=(0.75 * L, min(T_hot - 273.15) + 20),
    xytext=(0.25 * L, min(T_hot - 273.15) + 20),
    arrowprops=dict(arrowstyle="->", color="red", lw=2),
    color="red",
    fontsize=11
)

plt.annotate(
    "Cold flow",
    xy=(0.25 * L, min(T_cold - 273.15) - 10),
    xytext=(0.75 * L, min(T_cold - 273.15) - 10),
    arrowprops=dict(arrowstyle="->", color="blue", lw=2),
    color="blue",
    fontsize=11
)

if FLIP_PLOT_LEFT_RIGHT:
    plt.gca().invert_xaxis()
    xlabel = "Position x [m]  (left = L, right = 0)"
else:
    xlabel = "Position x [m]  (left = 0, right = L)"

plt.xlabel(xlabel, fontsize=12)
plt.ylabel("Temperature [°C]", fontsize=12)

plt.title(
    "Counter-flow Steam Generator Temperature Profile",
    fontsize=15,
    fontweight="bold"
)

plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()

plt.savefig(
    "steam_generator_temperature.png",
    dpi=300
)

plt.show()


# ============================================================
# Results
# ============================================================

print("\n==============================")
print("Steam Generator Results")
print("==============================")

print("[Direction]")
print("Hot water  : left  -> right")
print("Cold water : right -> left")

print("\n[Hot side]")
print(f"Hot inlet  at x = 0 : {T_hot[0] - 273.15:.2f} °C")
print(f"Hot outlet at x = L : {T_hot[-1] - 273.15:.2f} °C")

print("\n[Cold side]")
print(f"Cold inlet  at x = L : {T_cold[-1] - 273.15:.2f} °C")
print(f"Cold outlet at x = 0 : {T_cold[0] - 273.15:.2f} °C")

print("\n[Phase change]")
print(f"T_sat at P_cold = {P_cold / 1e6:.2f} MPa : {T_sat - 273.15:.2f} °C")
print(f"Latent heat H_fg : {H_fg / 1e3:.2f} kJ/kg")
print(f"Cold outlet quality at x = 0 : {quality[0]:.3f}")
print(f"Cold outlet phase : {phase[0]}")
print(f"Cold outlet superheat : {max(T_cold[0] - T_sat, 0.0):.2f} K")

print("\n[Correlation used in cold side]")
print(df_cell["cold_correlation"].value_counts())

if np.max(H_cold) <= H_g:
    print(
        "\nNote: Cold side did not reach the superheated vapor region. "
        "Increase L or m_dot_hot, raise T_hot_in, or reduce m_dot_cold "
        "if you want the gas-temperature-rise region to appear."
    )

print("==============================")
print("Node CSV saved: steam_generator_node_results.csv")
print("Cell CSV saved: steam_generator_cell_results.csv")
print("Figure saved: steam_generator_temperature.png")
