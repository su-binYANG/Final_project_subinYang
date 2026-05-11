# steam_generator_counterflow.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI


def dittus_boelter(Re, Pr, k, Dh, heating=True):
    n = 0.4 if heating else 0.3
    Nu = 0.023 * Re**0.8 * Pr**n
    h = Nu * k / Dh
    return h


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
        phase = "superheated"

        rho = PropsSI("D", "P", P_cold, "H", Hc, fluid_cold)
        mu = PropsSI("V", "P", P_cold, "H", Hc, fluid_cold)
        k = PropsSI("L", "P", P_cold, "H", Hc, fluid_cold)
        cp = PropsSI("C", "P", P_cold, "H", Hc, fluid_cold)

    return T, quality, phase, rho, mu, k, cp


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
T_cold_in = 420.0 + 273.15


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


# ============================================================
# Initial condition
# ============================================================

# Hot inlet: x = 0, left side
T_hot[0] = T_hot_in
H_hot[0] = PropsSI("H", "P", P_hot, "T", T_hot_in, fluid_hot)

# Cold inlet: x = L, right side
T_cold[N] = T_cold_in
H_cold[N] = PropsSI("H", "P", P_cold, "T", T_cold_in, fluid_cold)


# ============================================================
# Saturation properties for cold side
# ============================================================

T_sat = PropsSI("T", "P", P_cold, "Q", 0, fluid_cold)

H_f = PropsSI("H", "P", P_cold, "Q", 0, fluid_cold)
H_g = PropsSI("H", "P", P_cold, "Q", 1, fluid_cold)
H_fg = H_g - H_f


# ============================================================
# Main calculation
# ============================================================

for i in range(N):

    # Hot side index: left -> right
    ih = i

    # Cold side index: right -> left
    ic = N - i

    # -----------------------------
    # Hot side properties
    # -----------------------------
    rho_h = PropsSI("D", "P", P_hot, "H", H_hot[ih], fluid_hot)
    mu_h = PropsSI("V", "P", P_hot, "H", H_hot[ih], fluid_hot)
    k_h = PropsSI("L", "P", P_hot, "H", H_hot[ih], fluid_hot)
    cp_h = PropsSI("C", "P", P_hot, "H", H_hot[ih], fluid_hot)

    T_hot[ih] = PropsSI("T", "P", P_hot, "H", H_hot[ih], fluid_hot)

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

    # -----------------------------
    # Cold side properties
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

    h_cold = dittus_boelter(
        Re=Re_c,
        Pr=Pr_c,
        k=k_c,
        Dh=D_inner,
        heating=True
    )

    # -----------------------------
    # Overall heat transfer coefficient
    # -----------------------------
    U = 1.0 / (
        1.0 / h_hot
        + t_wall / k_wall
        + 1.0 / h_cold
    )

    # -----------------------------
    # Heat transfer
    # -----------------------------
    dT = T_hot[ih] - T_cold[ic]

    if dT < 0:
        q = 0.0
    else:
        q = U * P_heat * dx * dT

    # -----------------------------
    # Update hot side
    # Hot water loses heat
    # x = 0 -> L
    # -----------------------------
    H_hot[ih + 1] = H_hot[ih] - q / m_dot_hot

    # -----------------------------
    # Update cold side
    # Cold water gains heat
    # x = L -> 0
    # -----------------------------
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

df = pd.DataFrame({
    "x_m": x,
    "T_hot_C": T_hot - 273.15,
    "T_cold_C": T_cold - 273.15,
    "H_hot_Jkg": H_hot,
    "H_cold_Jkg": H_cold,
    "quality": quality,
    "cold_phase": phase
})

df.to_csv(
    "steam_generator_results.csv",
    index=False,
    encoding="utf-8-sig"
)

print(df)


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

plt.text(
    0.05 * L,
    T_hot[0] - 273.15 + 5,
    "Hot inlet",
    color="red"
)

plt.text(
    0.82 * L,
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

plt.xlabel("Position x [m]  (left = 0, right = L)", fontsize=12)
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
print(f"Cold outlet quality at x = 0 : {quality[0]:.3f}")
print(f"Cold outlet phase : {phase[0]}")

print("==============================")
print("CSV saved: steam_generator_results.csv")
print("Figure saved: steam_generator_temperature.png")
