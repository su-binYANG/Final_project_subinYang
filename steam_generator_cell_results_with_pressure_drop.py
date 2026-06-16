# steam_generator_counterflow_with_pressure_drop.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from functools import lru_cache
from CoolProp.CoolProp import PropsSI

FLIP_PLOT_LEFT_RIGHT = False


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


def del_col_dryout_incipience_quality(
    q_flux,
    G,
    Dh,
    h_fg,
    rho_l,
    rho_g,
    sigma,
    P,
    P_critical
):
    """
    Del Col et al. (2010) dryout incipience quality [-].
    """

    q_flux = max(q_flux, 1.0)
    G = max(G, 1.0e-12)
    Dh = max(Dh, 1.0e-12)
    h_fg = max(h_fg, 1.0e-12)
    rho_l = max(rho_l, 1.0e-12)
    rho_g = max(rho_g, 1.0e-12)
    sigma = max(sigma, 1.0e-12)
    P_critical = max(P_critical, 1.0e-12)

    Bo = q_flux / (G * h_fg)
    P_reduced = np.clip(P / P_critical, 1.0e-12, 0.999999)

    RLL = (
        0.437
        * (rho_g / rho_l)**0.073
        * (rho_l * sigma / G**2)**0.24
        * (Dh**0.721 / Bo)
    )**(1.0 / 0.96)

    x_di = (
        0.4695
        * (4.0 * q_flux * RLL / (G * Dh * h_fg))**1.472
        * (G**2 * Dh / (rho_l * sigma))**0.3024
        * (Dh / 0.001)**0.1836
        * (1.0 - P_reduced)**1.239
    )

    return x_di


def del_col_dryout_terms(
    q_flux,
    G,
    Dh,
    h_fg,
    rho_l,
    rho_g,
    sigma,
    P,
    P_critical
):
    """
    Del Col et al. (2010) dryout incipience quality and local terms.
    Units:
    q_flux [W/m2], G [kg/m2-s], Dh [m], h_fg [J/kg],
    rho_l/rho_g [kg/m3], sigma [N/m], pressures [Pa].
    """

    q_flux = max(q_flux, 1.0)
    G = max(G, 1.0e-12)
    Dh = max(Dh, 1.0e-12)
    h_fg = max(h_fg, 1.0e-12)
    rho_l = max(rho_l, 1.0e-12)
    rho_g = max(rho_g, 1.0e-12)
    sigma = max(sigma, 1.0e-12)
    P_critical = max(P_critical, 1.0e-12)

    Bo = q_flux / (G * h_fg)
    P_reduced = np.clip(P / P_critical, 1.0e-12, 0.999999)

    RLL = (
        0.437
        * (rho_g / rho_l)**0.073
        * (rho_l * sigma / G**2)**0.24
        * (Dh**0.721 / Bo)
    )**(1.0 / 0.96)

    x_di = (
        0.4695
        * (4.0 * q_flux * RLL / (G * Dh * h_fg))**1.472
        * (G**2 * Dh / (rho_l * sigma))**0.3024
        * (Dh / 0.001)**0.1836
        * (1.0 - P_reduced)**1.239
    )

    return {
        "x_di": x_di,
        "Bo": Bo,
        "RLL": RLL,
        "P_reduced": P_reduced,
        "rho_l": rho_l,
        "rho_g": rho_g,
        "sigma": sigma,
        "G": G,
        "h_fg": h_fg,
        "q_flux": q_flux,
    }


def dougall_rohsenow_post_chf_correlation(
    G,
    Dh,
    mu_g,
    rho_g,
    rho_l,
    Pr_g,
    k_g,
    quality
):
    """
    Dougall and Rohsenow (1963) post-CHF heat transfer coefficient [W/m2-K].
    """

    x = np.clip(quality, 1.0e-6, 0.999999)
    G = max(G, 1.0e-12)
    Dh = max(Dh, 1.0e-12)
    mu_g = max(mu_g, 1.0e-12)
    rho_g = max(rho_g, 1.0e-12)
    rho_l = max(rho_l, 1.0e-12)
    Pr_g = max(Pr_g, 1.0e-12)
    k_g = max(k_g, 1.0e-12)

    effective_re = (G * Dh / mu_g) * (x + (rho_g / rho_l) * (1.0 - x))

    return 0.023 * effective_re**0.8 * Pr_g**0.4 * (k_g / Dh)


def bergles_rohsenow_onb_deltaT(q_flux, P_cold):
    """
    Bergles and Rohsenow (1964) ONB wall superheat [K].

    P_cold is provided in Pa and converted to bar.
    q_flux is local heat flux [W/m2].
    """

    q_flux = max(q_flux, 1.0)
    P_bar = max(P_cold / 1.0e5, 1.0e-12)
    n = 0.463 * P_bar**0.0234
    deltaT_onb = 0.556 * (q_flux / (1082.0 * P_bar**1.156))**n

    return deltaT_onb


def subcooled_boiling_heat_transfer_coefficient(
    q_flux,
    G,
    D,
    h_fg,
    cp_l,
    k_l,
    mu_l_bulk,
    mu_l_wall,
    Re_l,
    Pr_l,
    T_sat,
    T_bulk
):
    """
    Subcooled boiling heat transfer coefficient [W/m2-K].

    h_tp = psi * h_sp_l
    psi = 267 * Bo**0.86 * Ja_star**(-0.6) * Pr_l**0.23
    """

    q_flux = max(q_flux, 1.0)
    G = max(G, 1.0e-12)
    D = max(D, 1.0e-12)
    h_fg = max(h_fg, 1.0e-12)
    cp_l = max(cp_l, 1.0e-12)
    k_l = max(k_l, 1.0e-12)
    mu_l_bulk = max(mu_l_bulk, 1.0e-12)
    mu_l_wall = max(mu_l_wall, 1.0e-12)
    Re_l = max(Re_l, 1.0)
    Pr_l = max(Pr_l, 1.0e-12)

    deltaT_sub_in = max(T_sat - T_bulk, 1.0e-6)
    Ja_star = cp_l * deltaT_sub_in / h_fg
    Bo = q_flux / (G * h_fg)

    h_sp_l = (
        0.023
        * Re_l**0.8
        * Pr_l**0.4
        * (mu_l_bulk / mu_l_wall)**0.262
        * k_l
        / D
    )
    psi = 267.0 * Bo**0.86 * Ja_star**(-0.6) * Pr_l**0.23

    return psi * h_sp_l


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

@lru_cache(maxsize=4096)
def _get_cold_saturation_properties_cached(P_cold_rounded, fluid_cold):
    P_cold = float(P_cold_rounded)
    T_sat = PropsSI("T", "P", P_cold, "Q", 0, fluid_cold)
    H_f = PropsSI("H", "P", P_cold, "Q", 0, fluid_cold)
    H_g = PropsSI("H", "P", P_cold, "Q", 1, fluid_cold)
    H_fg = H_g - H_f

    rho_l = PropsSI("D", "P", P_cold, "Q", 0, fluid_cold)
    rho_g = PropsSI("D", "P", P_cold, "Q", 1, fluid_cold)
    mu_l = PropsSI("V", "P", P_cold, "Q", 0, fluid_cold)
    mu_g = PropsSI("V", "P", P_cold, "Q", 1, fluid_cold)
    k_g = PropsSI("L", "P", P_cold, "Q", 1, fluid_cold)
    cp_g = PropsSI("C", "P", P_cold, "Q", 1, fluid_cold)
    Pr_g = cp_g * mu_g / k_g
    sigma = PropsSI("I", "P", P_cold, "Q", 0, fluid_cold)

    return {
        "T_sat": T_sat,
        "H_f": H_f,
        "H_g": H_g,
        "H_fg": H_fg,
        "rho_l": rho_l,
        "rho_g": rho_g,
        "mu_l": mu_l,
        "mu_g": mu_g,
        "k_g": k_g,
        "cp_g": cp_g,
        "Pr_g": Pr_g,
        "sigma": sigma,
    }


def get_cold_saturation_properties(P_cold, fluid_cold):
    P_cold_rounded = round(float(P_cold) / 1000.0) * 1000.0
    return _get_cold_saturation_properties_cached(P_cold_rounded, fluid_cold)


def get_cold_state(Hc, P_cold, fluid_cold, H_f, H_g, H_fg, T_sat):

    if Hc < H_f:
        T = PropsSI("T", "P", P_cold, "H", Hc, fluid_cold)
        quality = 0.0
        phase = "subcooled"

        rho = PropsSI("D", "P", P_cold, "H", Hc, fluid_cold)
        mu = PropsSI("V", "P", P_cold, "H", Hc, fluid_cold)
        k = PropsSI("L", "P", P_cold, "H", Hc, fluid_cold)
        cp = PropsSI("C", "P", P_cold, "H", Hc, fluid_cold)

    elif H_f <= Hc < H_g:
        T = T_sat
        quality = (Hc - H_f) / H_fg
        phase = "boiling"

        rho = PropsSI("D", "P", P_cold, "Q", 0, fluid_cold)
        mu = PropsSI("V", "P", P_cold, "Q", 0, fluid_cold)
        k = PropsSI("L", "P", P_cold, "Q", 0, fluid_cold)
        cp = PropsSI("C", "P", P_cold, "Q", 0, fluid_cold)

    else:
        quality = (Hc - H_f) / H_fg
        phase = "superheated_vapor"

        if Hc <= H_g:
            T = T_sat
            rho = PropsSI("D", "P", P_cold, "Q", 1, fluid_cold)
            mu = PropsSI("V", "P", P_cold, "Q", 1, fluid_cold)
            k = PropsSI("L", "P", P_cold, "Q", 1, fluid_cold)
            cp = PropsSI("C", "P", P_cold, "Q", 1, fluid_cold)
        else:
            T = PropsSI("T", "P", P_cold, "H", Hc, fluid_cold)
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

    elif phase == "subcooled_boiling":

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
            quality=1.0e-6
        )

        correlation = "Subcooled boiling after ONB"

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


def calc_wall_temperatures(
    T_hot_bulk,
    T_cold_effective,
    q_flux,
    h_hot,
    h_cold,
    wall_resistance_area=0.0
):
    """
    Local wall surface temperatures [K].

    q_flux is the local heat flux [W/m2]. The heat flows from hot fluid to
    cold fluid, so the hot-side wall is cooler than hot bulk fluid and the
    cold-side wall is hotter than cold bulk fluid.
    """

    h_hot = max(h_hot, 1.0e-12)
    h_cold = max(h_cold, 1.0e-12)

    q_flux = max(q_flux, 0.0)

    T_wall_hot = T_hot_bulk - q_flux / h_hot

    if wall_resistance_area > 0.0:
        T_wall_cold = T_wall_hot - q_flux * wall_resistance_area
    else:
        T_wall_cold = T_cold_effective + q_flux / h_cold

    return T_wall_hot, T_wall_cold


def get_cold_effective_temperature(phase_name, T_bulk, T_sat):
    if phase_name == "boiling":
        return T_sat

    return T_bulk


def cell_values_to_node_values(cell_values):
    """
    Convert cell-centered values to node values by endpoint copy and
    interior averaging.
    """

    node_values = np.zeros(N + 1)
    node_values[0] = cell_values[0]
    node_values[-1] = cell_values[-1]

    if N > 1:
        node_values[1:-1] = 0.5 * (cell_values[:-1] + cell_values[1:])

    return node_values


def save_csv_with_fallback(df, filename):
    try:
        df.to_csv(filename, index=False, encoding="utf-8-sig")
        return filename
    except PermissionError:
        stem, suffix = filename.rsplit(".", 1)
        fallback = f"{stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{suffix}"
        df.to_csv(fallback, index=False, encoding="utf-8-sig")
        print(f"Warning: {filename} is locked. Saved CSV as {fallback}")
        return fallback


# ============================================================
# Geometry and input conditions
# ============================================================

L = 51.852
N = 100
dx = L / N

D_inner = 0.012
D_outer = 0.016

A_flow = np.pi * D_inner**2 / 4.0
A_hot = 12.0 * A_flow
A_cold = A_flow
P_heat = np.pi * D_inner

t_wall = (D_outer - D_inner) / 2.0
k_wall = 16.0

fluid_hot = "Water"
fluid_cold = "Water"

P_hot_in = 15.0e6
T_hot_in = 600.0
u_hot_in = 2.0

P_cold_in = 6.0e6
T_cold_in = 530.0
u_cold_in = 2.0
T_cold_out_target = 563.15

P_hot = P_hot_in
P_cold = P_cold_in

rho_hot_in = PropsSI("D", "P", P_hot_in, "T", T_hot_in, fluid_hot)
rho_cold_in = PropsSI("D", "P", P_cold_in, "T", T_cold_in, fluid_cold)

m_dot_hot = rho_hot_in * A_hot * u_hot_in
m_dot_cold = rho_cold_in * A_cold * u_cold_in

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
h_single_phase_arr = np.zeros(N)
h_subcooled_arr = np.full(N, np.nan)
U_arr = np.zeros(N)
q_arr = np.zeros(N)
q_flux_arr = np.zeros(N)
T_hot_cell = np.zeros(N)
T_cold_cell = np.zeros(N)
quality_cell = np.zeros(N)
x_di_raw_cell = np.full(N, np.nan)
x_di_cell = np.full(N, np.nan)
Bo_cell = np.full(N, np.nan)
RLL_cell = np.full(N, np.nan)
P_reduced_cell = np.full(N, np.nan)
rho_l_cell = np.full(N, np.nan)
rho_g_cell = np.full(N, np.nan)
sigma_cell = np.full(N, np.nan)
G_cold_cell = np.full(N, np.nan)
h_fg_cell = np.full(N, np.nan)
CHF_region = ["pre_CHF"] * N
T_wall_hot_cell = np.zeros(N)
T_wall_cold_cell = np.zeros(N)
R_wall_cell = np.zeros(N)
T_sat_cell = np.zeros(N)
DeltaT_actual_cell = np.zeros(N)
DeltaT_ONB_cell = np.zeros(N)
cold_region = [""] * N

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
k_g_sat = PropsSI("L", "P", P_cold, "Q", 1, fluid_cold)
cp_g_sat = PropsSI("C", "P", P_cold, "Q", 1, fluid_cold)
Pr_g_sat = cp_g_sat * mu_g_sat / k_g_sat
sigma_sat = PropsSI("I", "P", P_cold, "Q", 0, fluid_cold)
P_critical_cold = PropsSI("Pcrit", fluid_cold)

G_hot = m_dot_hot / A_hot
G_cold = m_dot_cold / A_cold
x_onb = None
x_chf = None
x_superheated = None
H_hot_in = PropsSI("H", "P", P_hot, "T", T_hot_in, fluid_hot)
H_cold_in = PropsSI("H", "P", P_cold, "T", T_cold_in, fluid_cold)


# ============================================================
# Counter-current marching calculation
# ============================================================

def run_counter_current_march(H_hot_out_guess, P_hot_out_guess):
    """
    March from x = 0 to x = L in physical coordinates.

    Cold side flows with the marching direction, so its enthalpy increases from
    node 0 to node N. Hot side flows opposite to the marching direction; node 0
    is the unknown hot outlet, and node N must match the specified hot inlet.
    """

    T_hot_local = np.zeros(N + 1)
    H_hot_local = np.zeros(N + 1)
    T_cold_local = np.zeros(N + 1)
    H_cold_local = np.zeros(N + 1)
    P_hot_local = np.zeros(N + 1)
    P_cold_local = np.zeros(N + 1)
    quality_local = np.zeros(N + 1)
    phase_local = [""] * (N + 1)

    h_hot_local_arr = np.zeros(N)
    h_cold_local_arr = np.zeros(N)
    h_single_phase_local_arr = np.zeros(N)
    h_subcooled_local_arr = np.full(N, np.nan)
    U_local_arr = np.zeros(N)
    q_local_arr = np.zeros(N)
    q_flux_local_arr = np.zeros(N)
    P_hot_local_cell = np.zeros(N)
    P_cold_local_cell = np.zeros(N)
    T_hot_local_cell = np.zeros(N)
    T_cold_local_cell = np.zeros(N)
    quality_local_cell = np.zeros(N)
    x_di_raw_local_cell = np.full(N, np.nan)
    x_di_local_cell = np.full(N, np.nan)
    Bo_local_cell = np.full(N, np.nan)
    RLL_local_cell = np.full(N, np.nan)
    P_reduced_local_cell = np.full(N, np.nan)
    rho_l_local_cell = np.full(N, np.nan)
    rho_g_local_cell = np.full(N, np.nan)
    sigma_local_cell = np.full(N, np.nan)
    G_cold_local_cell = np.full(N, np.nan)
    h_fg_local_cell = np.full(N, np.nan)
    CHF_region_local = ["pre_CHF"] * N
    T_wall_hot_local_cell = np.zeros(N)
    T_wall_cold_local_cell = np.zeros(N)
    R_wall_local_cell = np.zeros(N)
    T_sat_local_cell = np.zeros(N)
    DeltaT_actual_local_cell = np.zeros(N)
    DeltaT_ONB_local_cell = np.zeros(N)
    cold_region_local = [""] * N
    dpdz_hot_local_arr = np.zeros(N)
    dpdz_cold_local_arr = np.zeros(N)
    dP_hot_local_cell = np.zeros(N)
    dP_cold_local_cell = np.zeros(N)
    phi2_local_arr = np.zeros(N)
    X_local_arr = np.zeros(N)
    cold_correlation_local = [""] * N
    pressure_correlation_local = [""] * N

    x_onb_local = None
    x_chf_local = None
    x_superheated_local = None
    chf_started_local = False

    H_hot_local[0] = H_hot_out_guess
    H_cold_local[0] = H_cold_in
    P_hot_local[0] = P_hot_out_guess
    P_cold_local[0] = P_cold_in

    for i in range(N):

        P_hot_i = P_hot_local[i]
        P_cold_i = P_cold_local[i]
        sat_i = get_cold_saturation_properties(P_cold_i, fluid_cold)
        T_sat_i = sat_i["T_sat"]
        H_f_i = sat_i["H_f"]
        H_g_i = sat_i["H_g"]
        H_fg_i = sat_i["H_fg"]
        rho_l_sat_i = sat_i["rho_l"]
        rho_g_sat_i = sat_i["rho_g"]
        mu_l_sat_i = sat_i["mu_l"]
        mu_g_sat_i = sat_i["mu_g"]
        k_g_sat_i = sat_i["k_g"]
        Pr_g_sat_i = sat_i["Pr_g"]
        sigma_sat_i = sat_i["sigma"]

        T_hot_local[i] = PropsSI("T", "P", P_hot_i, "H", H_hot_local[i], fluid_hot)

        rho_h = PropsSI("D", "P", P_hot_i, "H", H_hot_local[i], fluid_hot)
        mu_h = PropsSI("V", "P", P_hot_i, "H", H_hot_local[i], fluid_hot)
        k_h = PropsSI("L", "P", P_hot_i, "H", H_hot_local[i], fluid_hot)
        cp_h = PropsSI("C", "P", P_hot_i, "H", H_hot_local[i], fluid_hot)

        V_h = m_dot_hot / (rho_h * A_hot)
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

        (
            T_cold_local[i],
            quality_local[i],
            phase_local[i],
            rho_c,
            mu_c,
            k_c,
            cp_c
        ) = get_cold_state(
            H_cold_local[i],
            P_cold_i,
            fluid_cold,
            H_f_i,
            H_g_i,
            H_fg_i,
            T_sat_i
        )

        V_c = m_dot_cold / (rho_c * A_cold)
        Re_c = rho_c * V_c * D_inner / mu_c
        Pr_c = cp_c * mu_c / k_c
        h_single_phase = dittus_boelter(
            Re=Re_c,
            Pr=Pr_c,
            k=k_c,
            Dh=D_inner,
            heating=True
        )
        h_subcooled = np.nan
        T_cold_effective = get_cold_effective_temperature(
            phase_local[i],
            T_cold_local[i],
            T_sat_i
        )
        x_di_detection = np.nan
        x_di_raw_detection = np.nan
        del_col_terms_detection = None

        q_flux_guess = 50000.0

        h_cold, corr_name = get_cold_heat_transfer_coefficient(
            phase=phase_local[i],
            Re_c=Re_c,
            Pr_c=Pr_c,
            k_c=k_c,
            D_inner=D_inner,
            q_flux=q_flux_guess,
            G_cold=G_cold,
            H_fg=H_fg_i,
            quality_local=quality_local[i]
        )

        U = 1.0 / (
            1.0 / h_hot
            + t_wall / k_wall
            + 1.0 / h_cold
        )

        dT = T_hot_local[i] - T_cold_effective

        if dT <= 0.0:
            q = 0.0
            q_flux_local = 0.0

        else:
            q = U * P_heat * dx * dT
            q_flux_local = max(q / (P_heat * dx), 0.0)

            if phase_local[i] == "boiling":

                h_cold, corr_name = get_cold_heat_transfer_coefficient(
                    phase=phase_local[i],
                    Re_c=Re_c,
                    Pr_c=Pr_c,
                    k_c=k_c,
                    D_inner=D_inner,
                    q_flux=q_flux_local,
                    G_cold=G_cold,
                    H_fg=H_fg_i,
                    quality_local=quality_local[i]
                )

                U = 1.0 / (
                    1.0 / h_hot
                    + t_wall / k_wall
                    + 1.0 / h_cold
                )

                q = U * P_heat * dx * dT
                q_flux_local = max(q / (P_heat * dx), 0.0)

            if phase_local[i] == "boiling":

                del_col_terms_detection = del_col_dryout_terms(
                    q_flux=q_flux_local,
                    G=G_cold,
                    Dh=D_inner,
                    h_fg=H_fg_i,
                    rho_l=rho_l_sat_i,
                    rho_g=rho_g_sat_i,
                    sigma=sigma_sat_i,
                    P=P_cold_i,
                    P_critical=P_critical_cold
                )
                x_di_detection = del_col_terms_detection["x_di"]
                x_di_raw_detection = x_di_detection

                is_post_chf = chf_started_local or quality_local[i] >= x_di_detection

                if is_post_chf:
                    h_cold = dougall_rohsenow_post_chf_correlation(
                        G=G_cold,
                        Dh=D_inner,
                        mu_g=mu_g_sat_i,
                        rho_g=rho_g_sat_i,
                        rho_l=rho_l_sat_i,
                        Pr_g=Pr_g_sat_i,
                        k_g=k_g_sat_i,
                        quality=quality_local[i]
                    )
                    corr_name = "Dougall-Rohsenow post-CHF"

                    U = 1.0 / (
                        1.0 / h_hot
                        + t_wall / k_wall
                        + 1.0 / h_cold
                    )

                    q = U * P_heat * dx * dT
                    q_flux_local = max(q / (P_heat * dx), 0.0)

        T_wall_hot_local, T_wall_cold_local = calc_wall_temperatures(
            T_hot_bulk=T_hot_local[i],
            T_cold_effective=T_cold_effective,
            q_flux=q_flux_local,
            h_hot=h_hot,
            h_cold=h_cold,
            wall_resistance_area=t_wall / k_wall
        )
        DeltaT_actual = T_wall_cold_local - T_sat_i
        DeltaT_ONB = bergles_rohsenow_onb_deltaT(q_flux_local, P_cold_i)
        x_di_local = x_di_detection
        x_di_raw_local = x_di_raw_detection
        del_col_terms_local = del_col_terms_detection
        chf_region_name = "post_CHF" if chf_started_local else "pre_CHF"

        if phase_local[i] == "boiling":
            if np.isnan(x_di_local):
                del_col_terms_local = del_col_dryout_terms(
                    q_flux=q_flux_local,
                    G=G_cold,
                    Dh=D_inner,
                    h_fg=H_fg_i,
                    rho_l=rho_l_sat_i,
                    rho_g=rho_g_sat_i,
                    sigma=sigma_sat_i,
                    P=P_cold_i,
                    P_critical=P_critical_cold
                )
                x_di_local = del_col_terms_local["x_di"]
                x_di_raw_local = x_di_local

            if chf_started_local or quality_local[i] >= x_di_local:
                chf_region_name = "post_CHF"
                if not chf_started_local:
                    chf_started_local = True
                    x_chf_local = x[i]
            else:
                chf_region_name = "pre_CHF"

        if phase_local[i] == "subcooled":
            is_onb = (
                T_cold_local[i] < T_sat_i
                and DeltaT_actual >= DeltaT_ONB
            )

            if x_onb_local is None and is_onb:
                x_onb_local = x[i]

            if is_onb:
                for _ in range(3):
                    if T_wall_cold_local >= T_sat_i - 0.1:
                        mu_l_wall = mu_l_sat_i
                    else:
                        mu_l_wall = PropsSI(
                            "V",
                            "P",
                            P_cold_i,
                            "T",
                            T_wall_cold_local,
                            fluid_cold
                        )
                    h_subcooled = subcooled_boiling_heat_transfer_coefficient(
                        q_flux=q_flux_local,
                        G=G_cold,
                        D=D_inner,
                        h_fg=H_fg_i,
                        cp_l=cp_c,
                        k_l=k_c,
                        mu_l_bulk=mu_c,
                        mu_l_wall=mu_l_wall,
                        Re_l=Re_c,
                        Pr_l=Pr_c,
                        T_sat=T_sat_i,
                        T_bulk=T_cold_local[i]
                    )
                    h_cold = h_subcooled
                    corr_name = "Subcooled boiling h_tp = psi*h_sp_l"

                    U = 1.0 / (
                        1.0 / h_hot
                        + t_wall / k_wall
                        + 1.0 / h_cold
                    )

                    if dT <= 0.0:
                        q = 0.0
                        q_flux_local = 0.0
                    else:
                        q = U * P_heat * dx * dT
                        q_flux_local = max(q / (P_heat * dx), 0.0)

                    T_wall_hot_local, T_wall_cold_local = calc_wall_temperatures(
                        T_hot_bulk=T_hot_local[i],
                        T_cold_effective=T_cold_effective,
                        q_flux=q_flux_local,
                        h_hot=h_hot,
                        h_cold=h_cold,
                        wall_resistance_area=t_wall / k_wall
                    )

                DeltaT_actual = T_wall_cold_local - T_sat_i
                DeltaT_ONB = bergles_rohsenow_onb_deltaT(q_flux_local, P_cold_i)
                region_name = "Subcooled Boiling"
            else:
                region_name = "Single Phase Liquid"
        elif phase_local[i] == "boiling":
            if chf_region_name == "post_CHF":
                region_name = "CHF / Dryout"
            else:
                region_name = "Saturated Boiling"
        elif phase_local[i] == "superheated_vapor":
            region_name = "Superheated Steam"
            if x_superheated_local is None:
                x_superheated_local = x[i]
        else:
            region_name = phase_local[i]

        if phase_local[i] == "boiling":

            dpdz_cold, phi2, X, Re_l, Re_g = separated_flow_pressure_drop(
                quality=quality_local[i],
                G=G_cold,
                Dh=D_inner,
                rho_l=rho_l_sat_i,
                rho_g=rho_g_sat_i,
                mu_l=mu_l_sat_i,
                mu_g=mu_g_sat_i,
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

        T_wall_hot_local, T_wall_cold_local = calc_wall_temperatures(
            T_hot_bulk=T_hot_local[i],
            T_cold_effective=T_cold_effective,
            q_flux=q_flux_local,
            h_hot=h_hot,
            h_cold=h_cold,
            wall_resistance_area=t_wall / k_wall
        )

        h_hot_local_arr[i] = h_hot
        h_cold_local_arr[i] = h_cold
        h_single_phase_local_arr[i] = h_single_phase
        h_subcooled_local_arr[i] = h_subcooled
        U_local_arr[i] = U
        q_local_arr[i] = q
        q_flux_local_arr[i] = q_flux_local
        P_hot_local_cell[i] = P_hot_i
        P_cold_local_cell[i] = P_cold_i
        T_hot_local_cell[i] = T_hot_local[i]
        T_cold_local_cell[i] = T_cold_local[i]
        quality_local_cell[i] = quality_local[i]
        x_di_raw_local_cell[i] = x_di_raw_local
        x_di_local_cell[i] = x_di_local
        if del_col_terms_local is not None:
            Bo_local_cell[i] = del_col_terms_local["Bo"]
            RLL_local_cell[i] = del_col_terms_local["RLL"]
            P_reduced_local_cell[i] = del_col_terms_local["P_reduced"]
            rho_l_local_cell[i] = del_col_terms_local["rho_l"]
            rho_g_local_cell[i] = del_col_terms_local["rho_g"]
            sigma_local_cell[i] = del_col_terms_local["sigma"]
            G_cold_local_cell[i] = del_col_terms_local["G"]
            h_fg_local_cell[i] = del_col_terms_local["h_fg"]
        CHF_region_local[i] = chf_region_name
        T_wall_hot_local_cell[i] = T_wall_hot_local
        T_wall_cold_local_cell[i] = T_wall_cold_local
        R_wall_local_cell[i] = np.log(D_outer / D_inner) / (2.0 * np.pi * k_wall * dx)
        T_sat_local_cell[i] = T_sat_i
        DeltaT_actual_local_cell[i] = DeltaT_actual
        DeltaT_ONB_local_cell[i] = DeltaT_ONB
        cold_region_local[i] = region_name

        dpdz_hot_local_arr[i] = dpdz_hot
        dpdz_cold_local_arr[i] = dpdz_cold
        dP_hot_local_cell[i] = dpdz_hot * dx
        dP_cold_local_cell[i] = dpdz_cold * dx
        phi2_local_arr[i] = phi2
        X_local_arr[i] = X
        cold_correlation_local[i] = corr_name
        pressure_correlation_local[i] = pressure_corr

        # Moving from x = 0 to x = L is opposite to the hot-flow direction.
        H_hot_local[i + 1] = H_hot_local[i] + q / m_dot_hot
        H_cold_local[i + 1] = H_cold_local[i] + q / m_dot_cold
        P_hot_local[i + 1] = P_hot_local[i] + dP_hot_local_cell[i]
        P_cold_local[i + 1] = max(P_cold_local[i] - dP_cold_local_cell[i], 1.0e5)

    for i in range(N + 1):
        sat_i = get_cold_saturation_properties(P_cold_local[i], fluid_cold)
        T_hot_local[i] = PropsSI("T", "P", P_hot_local[i], "H", H_hot_local[i], fluid_hot)
        (
            T_cold_local[i],
            quality_local[i],
            phase_local[i],
            rho_c,
            mu_c,
            k_c,
            cp_c
        ) = get_cold_state(
            H_cold_local[i],
            P_cold_local[i],
            fluid_cold,
            sat_i["H_f"],
            sat_i["H_g"],
            sat_i["H_fg"],
            sat_i["T_sat"]
        )

    if x_superheated_local is None:
        superheated_node_indices = np.where(np.array(phase_local) == "superheated_vapor")[0]
        if superheated_node_indices.size > 0:
            x_superheated_local = x[superheated_node_indices[0]]

    return {
        "T_hot": T_hot_local,
        "H_hot": H_hot_local,
        "T_cold": T_cold_local,
        "H_cold": H_cold_local,
        "P_hot": P_hot_local,
        "P_cold": P_cold_local,
        "quality": quality_local,
        "phase": phase_local,
        "h_hot_arr": h_hot_local_arr,
        "h_cold_arr": h_cold_local_arr,
        "h_single_phase_arr": h_single_phase_local_arr,
        "h_subcooled_arr": h_subcooled_local_arr,
        "U_arr": U_local_arr,
        "q_arr": q_local_arr,
        "q_flux_arr": q_flux_local_arr,
        "P_hot_cell": P_hot_local_cell,
        "P_cold_cell": P_cold_local_cell,
        "T_hot_cell": T_hot_local_cell,
        "T_cold_cell": T_cold_local_cell,
        "quality_cell": quality_local_cell,
        "x_di_raw_cell": x_di_raw_local_cell,
        "x_di_cell": x_di_local_cell,
        "Bo_cell": Bo_local_cell,
        "RLL_cell": RLL_local_cell,
        "P_reduced_cell": P_reduced_local_cell,
        "rho_l_cell": rho_l_local_cell,
        "rho_g_cell": rho_g_local_cell,
        "sigma_cell": sigma_local_cell,
        "G_cold_cell": G_cold_local_cell,
        "h_fg_cell": h_fg_local_cell,
        "CHF_region": CHF_region_local,
        "T_wall_hot_cell": T_wall_hot_local_cell,
        "T_wall_cold_cell": T_wall_cold_local_cell,
        "R_wall_cell": R_wall_local_cell,
        "T_sat_cell": T_sat_local_cell,
        "DeltaT_actual_cell": DeltaT_actual_local_cell,
        "DeltaT_ONB_cell": DeltaT_ONB_local_cell,
        "cold_region": cold_region_local,
        "dpdz_hot_arr": dpdz_hot_local_arr,
        "dpdz_cold_arr": dpdz_cold_local_arr,
        "dP_hot_cell": dP_hot_local_cell,
        "dP_cold_cell": dP_cold_local_cell,
        "phi2_arr": phi2_local_arr,
        "X_arr": X_local_arr,
        "cold_correlation": cold_correlation_local,
        "pressure_correlation": pressure_correlation_local,
        "x_onb": x_onb_local,
        "x_chf": x_chf_local,
        "x_superheated": x_superheated_local,
    }


def solve_hot_outlet_enthalpy(P_hot_out_guess):
    T_low = max(PropsSI("Tmin", fluid_hot) + 1.0, T_cold_in)
    T_high = T_hot_in
    H_low = PropsSI("H", "P", P_hot_out_guess, "T", T_low, fluid_hot)
    H_high = H_hot_in

    def residual(H_guess):
        return run_counter_current_march(H_guess, P_hot_out_guess)["H_hot"][-1] - H_hot_in

    r_low = residual(H_low)
    r_high = residual(H_high)

    if r_low * r_high > 0.0:
        print("Warning: could not bracket hot outlet enthalpy for counter-current solve")
        return H_low if abs(r_low) < abs(r_high) else H_high

    for _ in range(10):
        H_mid = 0.5 * (H_low + H_high)
        r_mid = residual(H_mid)

        if abs(r_mid) < 50.0:
            return H_mid

        if r_low * r_mid <= 0.0:
            H_high = H_mid
            r_high = r_mid
        else:
            H_low = H_mid
            r_low = r_mid

    return 0.5 * (H_low + H_high)


def solve_hot_outlet_conditions():
    P_hot_out_guess = P_hot_in
    H_hot_out_guess = H_hot_in

    for _ in range(3):
        H_hot_out_guess = solve_hot_outlet_enthalpy(P_hot_out_guess)
        trial = run_counter_current_march(H_hot_out_guess, P_hot_out_guess)
        pressure_residual = trial["P_hot"][-1] - P_hot_in

        if abs(pressure_residual) < 100.0:
            return H_hot_out_guess, P_hot_out_guess, trial

        P_hot_out_guess = max(P_hot_out_guess - pressure_residual, 1.0e5)

    return H_hot_out_guess, P_hot_out_guess, run_counter_current_march(
        H_hot_out_guess,
        P_hot_out_guess
    )


H_hot_out, P_hot_out, solution = solve_hot_outlet_conditions()

T_hot = solution["T_hot"]
H_hot = solution["H_hot"]
T_cold = solution["T_cold"]
H_cold = solution["H_cold"]
P_hot_node = solution["P_hot"]
P_cold_node = solution["P_cold"]
quality = solution["quality"]
phase = solution["phase"]
h_hot_arr = solution["h_hot_arr"]
h_cold_arr = solution["h_cold_arr"]
h_single_phase_arr = solution["h_single_phase_arr"]
h_subcooled_arr = solution["h_subcooled_arr"]
U_arr = solution["U_arr"]
q_arr = solution["q_arr"]
q_flux_arr = solution["q_flux_arr"]
P_hot_cell = solution["P_hot_cell"]
P_cold_cell = solution["P_cold_cell"]
T_hot_cell = solution["T_hot_cell"]
T_cold_cell = solution["T_cold_cell"]
quality_cell = solution["quality_cell"]
x_di_raw_cell = solution["x_di_raw_cell"]
x_di_cell = solution["x_di_cell"]
Bo_cell = solution["Bo_cell"]
RLL_cell = solution["RLL_cell"]
P_reduced_cell = solution["P_reduced_cell"]
rho_l_cell = solution["rho_l_cell"]
rho_g_cell = solution["rho_g_cell"]
sigma_cell = solution["sigma_cell"]
G_cold_cell = solution["G_cold_cell"]
h_fg_cell = solution["h_fg_cell"]
CHF_region = solution["CHF_region"]
T_wall_hot_cell = solution["T_wall_hot_cell"]
T_wall_cold_cell = solution["T_wall_cold_cell"]
R_wall_cell = solution["R_wall_cell"]
T_sat_cell = solution["T_sat_cell"]
DeltaT_actual_cell = solution["DeltaT_actual_cell"]
DeltaT_ONB_cell = solution["DeltaT_ONB_cell"]
cold_region = solution["cold_region"]
dpdz_hot_arr = solution["dpdz_hot_arr"]
dpdz_cold_arr = solution["dpdz_cold_arr"]
dP_hot_cell = solution["dP_hot_cell"]
dP_cold_cell = solution["dP_cold_cell"]
phi2_arr = solution["phi2_arr"]
X_arr = solution["X_arr"]
cold_correlation = solution["cold_correlation"]
pressure_correlation = solution["pressure_correlation"]
x_onb = solution["x_onb"]
x_chf = solution["x_chf"]
x_superheated = solution["x_superheated"]


# ============================================================
# Final node state conversion
# ============================================================

for i in range(N + 1):

    sat_i = get_cold_saturation_properties(P_cold_node[i], fluid_cold)
    T_hot[i] = PropsSI("T", "P", P_hot_node[i], "H", H_hot[i], fluid_hot)

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
        P_cold_node[i],
        fluid_cold,
        sat_i["H_f"],
        sat_i["H_g"],
        sat_i["H_fg"],
        sat_i["T_sat"]
    )

if x_superheated is None:
    superheated_node_indices = np.where(np.array(phase) == "superheated_vapor")[0]
    if superheated_node_indices.size > 0:
        x_superheated = x[superheated_node_indices[0]]

T_hot_local = cell_values_to_node_values(T_hot_cell)
T_cold_local = cell_values_to_node_values(T_cold_cell)
T_wall_hot = cell_values_to_node_values(T_wall_hot_cell)
T_wall_cold = cell_values_to_node_values(T_wall_cold_cell)
T_sat_node = np.array([
    get_cold_saturation_properties(P_cold_node[i], fluid_cold)["T_sat"]
    for i in range(N + 1)
])
DeltaT_actual_node = T_wall_cold - T_sat_node
DeltaT_ONB_node = cell_values_to_node_values(DeltaT_ONB_cell)
ONB_margin_cell = DeltaT_actual_cell - DeltaT_ONB_cell
ONB_margin_node = DeltaT_actual_node - DeltaT_ONB_node
q_flux_node = cell_values_to_node_values(q_flux_arr)
h_cold_node = cell_values_to_node_values(h_cold_arr)
h_single_phase_node = cell_values_to_node_values(h_single_phase_arr)
h_subcooled_node = cell_values_to_node_values(h_subcooled_arr)
cold_region_node = [""] * (N + 1)
for i in range(N + 1):
    if i == 0:
        cold_region_node[i] = cold_region[0]
    elif i == N:
        cold_region_node[i] = cold_region[-1]
    else:
        cold_region_node[i] = cold_region[i - 1]

subcooled_onb_mask = T_cold_cell < T_sat_cell
if np.any(subcooled_onb_mask):
    onb_debug_indices = np.where(subcooled_onb_mask)[0]
    best_onb_idx = onb_debug_indices[np.argmax(ONB_margin_cell[onb_debug_indices])]
else:
    best_onb_idx = int(np.argmax(ONB_margin_cell))

max_deltaT_actual = float(np.max(DeltaT_actual_cell))
min_deltaT_ONB = float(np.min(DeltaT_ONB_cell))
max_ONB_margin = float(ONB_margin_cell[best_onb_idx])
x_max_ONB_margin = float(0.5 * (x[best_onb_idx] + x[best_onb_idx + 1]))

if np.any(T_wall_cold > T_wall_hot):
    print("Warning: cold-side wall temperature exceeds hot-side wall temperature")

if np.any(T_wall_hot > T_hot_local):
    print("Warning: hot-side wall temperature exceeds hot fluid temperature")

if np.any(T_wall_cold < T_cold_local):
    print("Warning: cold-side wall temperature lower than cold fluid temperature")

if np.any(T_wall_hot_cell > T_hot_cell):
    print("Warning: cell hot-side wall temperature exceeds hot fluid temperature")

if np.any(T_wall_cold_cell < T_cold_cell):
    print("Warning: cell cold-side wall temperature lower than cold fluid temperature")

if np.any(T_wall_hot_cell < T_wall_cold_cell):
    print("Warning: cell cold-side wall temperature exceeds hot-side wall temperature")

if np.any(T_wall_cold_cell > T_wall_hot_cell):
    print("Warning: cold-side wall temperature exceeds hot-side wall temperature")

if np.any(T_wall_cold_cell > T_hot_cell):
    print("Warning: cold-side wall temperature exceeds hot fluid temperature")

if np.any(T_wall_cold_cell < T_cold_cell):
    print("Warning: cold-side wall temperature lower than cold fluid temperature")


# ============================================================
# Cumulative pressure drop
# Cold water flows from x = 0 to x = L.
# Therefore cumulative dP is accumulated from left to right.
# ============================================================

dP_cold_cumulative[0] = 0.0

for i in range(N):
    dP_cold_cumulative[i + 1] = dP_cold_cumulative[i] + dP_cold_cell[i]

dP_cold_cumulative_cell[:] = dP_cold_cumulative[:-1]


# ============================================================
# Dataframes
# ============================================================

df_node = pd.DataFrame({
    "x_m": x,
    "P_hot": P_hot_node,
    "P_cold": P_cold_node,
    "T_hot_C": T_hot - 273.15,
    "T_cold": T_cold,
    "T_cold_C": T_cold - 273.15,
    "T_wall_hot_C": T_wall_hot - 273.15,
    "T_wall_cold": T_wall_cold,
    "T_wall_cold_C": T_wall_cold - 273.15,
    "T_sat": T_sat_node,
    "T_sat_cold_C": T_sat_node - 273.15,
    "DeltaT_actual": DeltaT_actual_node,
    "DeltaT_ONB": DeltaT_ONB_node,
    "ONB_margin": ONB_margin_node,
    "q_flux": q_flux_node,
    "h_single_phase": h_single_phase_node,
    "h_subcooled": h_subcooled_node,
    "h_cold": h_cold_node,
    "cold_region": cold_region_node,
    "cold_quality": quality,
    "cold_phase": phase,
    "cold_cumulative_dP_Pa": dP_cold_cumulative,
    "cold_cumulative_dP_kPa": dP_cold_cumulative / 1000.0,
})

df_cell = pd.DataFrame({
    "cell": np.arange(N),
    "x_position": 0.5 * (x[:-1] + x[1:]),
    "x_mid_m": 0.5 * (x[:-1] + x[1:]),
    "T_hot": T_hot_cell,
    "T_cold": T_cold_cell,
    "T_sat": T_sat_cell,
    "T_wall_hot": T_wall_hot_cell,
    "T_wall_cold": T_wall_cold_cell,
    "DeltaT_actual": DeltaT_actual_cell,
    "DeltaT_ONB": DeltaT_ONB_cell,
    "ONB_margin": ONB_margin_cell,
    "quality": quality_cell,
    "Bo": Bo_cell,
    "RLL": RLL_cell,
    "P_reduced": P_reduced_cell,
    "rho_l": rho_l_cell,
    "rho_g": rho_g_cell,
    "sigma": sigma_cell,
    "G": G_cold_cell,
    "h_fg": h_fg_cell,
    "q_flux": q_flux_arr,
    "x_di_raw": x_di_raw_cell,
    "x_di": x_di_cell,
    "cold_region": cold_region,
    "h_single_phase": h_single_phase_arr,
    "h_subcooled": h_subcooled_arr,
    "h_hot": h_hot_arr,
    "h_cold": h_cold_arr,
    "P_hot": P_hot_cell,
    "P_cold": P_cold_cell,
    "h_hot_W_m2K": h_hot_arr,
    "h_cold_W_m2K": h_cold_arr,
    "U_W_m2K": U_arr,
    "q_W": q_arr,
    "q_flux_W_m2": q_flux_arr,
    "T_hot_C": T_hot_cell - 273.15,
    "T_cold_C": T_cold_cell - 273.15,
    "x": quality_cell,
    "CHF_region": CHF_region,
    "T_wall_hot_C": T_wall_hot_cell - 273.15,
    "T_wall_cold_C": T_wall_cold_cell - 273.15,
    "T_sat_C": T_sat_cell - 273.15,
    "T_wall": T_wall_cold_cell - 273.15,
    "R_wall_K_W": R_wall_cell,
    "R_wall_area_m2K_W": np.full(N, t_wall / k_wall),
    "wall_deltaT_conduction_C": q_flux_arr * (t_wall / k_wall),
    "dpdz_hot_Pa_m": dpdz_hot_arr,
    "dpdz_cold_Pa_m": dpdz_cold_arr,
    "dP_hot_cell_Pa": dP_hot_cell,
    "dP_cold_cell_Pa": dP_cold_cell,
    "cold_cumulative_dP_Pa": dP_cold_cumulative_cell,
    "cold_cumulative_dP_kPa": dP_cold_cumulative_cell / 1000.0,
    "phi2": phi2_arr,
    "Martinelli_X": X_arr,
    "cold_heat_transfer_correlation": cold_correlation,
    "pressure_drop_correlation": pressure_correlation
})

node_csv_file = save_csv_with_fallback(
    df_node,
    "steam_generator_node_results_with_pressure_drop.csv"
)

cell_csv_file = save_csv_with_fallback(
    df_cell,
    "steam_generator_cell_results_with_pressure_drop.csv"
)


# ============================================================
# Plot : temperature + pressure drop
# ============================================================

fig, ax1 = plt.subplots(figsize=(15, 6.8))

chf_x = None
dryout_x = None

ax1.plot(
    x,
    T_hot_local - 273.15,
    color="red",
    linestyle="-",
    linewidth=3,
    marker=None,
    label="Hot water temperature"
)

ax1.plot(
    x,
    T_cold_local - 273.15,
    color="blue",
    linestyle="-",
    linewidth=3,
    marker=None,
    label="Cold water / steam temperature"
)

ax1.plot(
    x,
    T_wall_hot - 273.15,
    color="orange",
    linestyle="-",
    linewidth=3,
    marker=None,
    label="Hot-side wall temperature"
)

ax1.plot(
    x,
    T_wall_cold - 273.15,
    color="green",
    linestyle="-",
    linewidth=3,
    marker=None,
    label="Cold-side wall temperature"
)

ax1.axhline(
    T_sat_node[0] - 273.15,
    linestyle="--",
    linewidth=1.5,
    alpha=0.0,
    label="_hidden_constant_sat_reference"
)

ax1.plot(
    x,
    T_sat_node - 273.15,
    color="gold",
    linestyle="-",
    linewidth=2.5,
    marker=None,
    label="Cold saturation temperature"
)

region_styles = {
    "Single Phase Liquid": {"color": "tab:blue", "alpha": 0.08},
    "Subcooled Boiling": {
        "color": "tab:green",
        "alpha": 0.24,
        "edgecolor": "tab:green",
        "hatch": "///",
        "linewidth": 1.2,
    },
    "Saturated Boiling": {"color": "tab:cyan", "alpha": 0.14},
    "CHF / Dryout": {"color": "tab:red", "alpha": 0.12},
    "Superheated Steam": {"color": "tab:purple", "alpha": 0.12},
}

shown_region_labels = set()
min_visible_region_width = 0.02 * L
start_idx = 0
while start_idx < N:
    region = cold_region[start_idx]
    end_idx = start_idx + 1
    while end_idx < N and cold_region[end_idx] == region:
        end_idx += 1

    style = region_styles.get(region)
    if style is not None:
        label = region if region not in shown_region_labels else None
        span_left = x[start_idx]
        span_right = x[end_idx]

        if region == "Subcooled Boiling":
            span_center = 0.5 * (span_left + span_right)
            visible_width = max(span_right - span_left, min_visible_region_width)
            span_left = max(x[0], span_center - 0.5 * visible_width)
            span_right = min(x[-1], span_center + 0.5 * visible_width)

        ax1.axvspan(
            span_left,
            span_right,
            facecolor=style["color"],
            alpha=style["alpha"],
            edgecolor=style.get("edgecolor", None),
            hatch=style.get("hatch", None),
            linewidth=style.get("linewidth", 0.0),
            label=label
        )
        shown_region_labels.add(region)

    start_idx = end_idx

if x_onb is not None:
    ax1.axvline(
        x_onb,
        color="tab:orange",
        linestyle="--",
        linewidth=2,
        label="ONB start"
    )

if chf_x is not None:
    ax1.axvline(
        chf_x,
        color="tab:red",
        linestyle="--",
        linewidth=2,
        label="CHF location"
    )

if x_chf is not None:
    ax1.axvline(
        x_chf,
        color="red",
        linestyle="--",
        linewidth=2,
        label="CHF start"
    )

if x_superheated is not None:
    ax1.axvline(
        x_superheated,
        color="tab:purple",
        linestyle="--",
        linewidth=2,
        label="Superheated Steam start"
    )

if dryout_x is not None:
    ax1.axvline(
        dryout_x,
        color="tab:purple",
        linestyle="--",
        linewidth=2,
        label="Dryout start"
    )

if FLIP_PLOT_LEFT_RIGHT:
    ax1.invert_xaxis()
    xlabel = "Position x [m]  (left = L, right = 0)"
else:
    xlabel = "Position x [m]  (left = 0, right = L)"

ax1.set_xlabel(xlabel)
ax1.set_ylabel("Temperature [degC]")
ax1.grid(True, linestyle="--", alpha=0.5)

arrow_y = 0.96

ax1.annotate(
    "Hot water flow: x = L -> x = 0",
    xy=(0.58, arrow_y),
    xytext=(0.92, arrow_y),
    xycoords="axes fraction",
    textcoords="axes fraction",
    arrowprops={"arrowstyle": "->", "color": "tab:red", "lw": 1.8},
    color="tab:red",
    fontsize=10,
    ha="center",
    va="center"
)

ax1.annotate(
    "Cold water flow: x = 0 -> x = L",
    xy=(0.42, arrow_y - 0.06),
    xytext=(0.08, arrow_y - 0.06),
    xycoords="axes fraction",
    textcoords="axes fraction",
    arrowprops={"arrowstyle": "->", "color": "tab:blue", "lw": 1.8},
    color="tab:blue",
    fontsize=10,
    ha="center",
    va="center"
)

ax2 = ax1.twinx()

ax2.plot(
    x,
    dP_cold_cumulative / 1000.0,
    linewidth=2,
    linestyle="--",
    label="Cold-side cumulative pressure drop"
)

ax2.tick_params(axis="y", pad=8)
ax2.set_ylabel("Cold-side cumulative pressure drop [kPa]", labelpad=14)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
legend_order = [
    "Hot water temperature",
    "Hot-side wall temperature",
    "Cold water / steam temperature",
    "Cold-side wall temperature",
    "Cold saturation temperature",
    "Single Phase Liquid",
    "Subcooled Boiling",
    "Saturated Boiling",
    "CHF / Dryout",
    "Superheated Steam",
    "ONB start",
    "CHF start",
    "Superheated Steam start",
    "Cold-side cumulative pressure drop",
]
legend_items = {
    label: handle
    for handle, label in zip(lines1 + lines2, labels1 + labels2)
    if not label.startswith("_")
}
ordered_labels = [label for label in legend_order if label in legend_items]

ax1.legend(
    [legend_items[label] for label in ordered_labels],
    ordered_labels,
    loc="upper left",
    bbox_to_anchor=(1.18, 1.0),
    borderaxespad=0,
    frameon=True
)

ax1.set_title(
    "Counter-flow Steam Generator: Temperature and Pressure Drop",
    fontsize=15,
    fontweight="bold",
    pad=18
)

fig.subplots_adjust(left=0.08, right=0.64, top=0.86, bottom=0.13)
plt.savefig(
    "steam_generator_temperature_pressure_drop.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close(fig)

fig_onb, ax_onb = plt.subplots(figsize=(12, 4.8))
x_mid = 0.5 * (x[:-1] + x[1:])

ax_onb.plot(
    x_mid,
    DeltaT_actual_cell,
    linewidth=2.6,
    marker="o",
    markersize=3,
    label="DeltaT_actual = T_wall_cold - T_sat"
)
ax_onb.plot(
    x_mid,
    DeltaT_ONB_cell,
    linewidth=2.6,
    marker="s",
    markersize=3,
    label="DeltaT_ONB: Bergles-Rohsenow"
)
ax_onb.plot(
    x_mid,
    ONB_margin_cell,
    linewidth=2.0,
    linestyle="--",
    color="tab:gray",
    label="ONB_margin = DeltaT_actual - DeltaT_ONB"
)
ax_onb.axhline(
    0.0,
    color="black",
    linestyle=":",
    linewidth=1.3
)

if x_onb is not None:
    ax_onb.axvline(
        x_onb,
        color="tab:orange",
        linestyle=":",
        linewidth=2.4,
        label="ONB start"
    )

if FLIP_PLOT_LEFT_RIGHT:
    ax_onb.invert_xaxis()

ax_onb.set_xlabel(xlabel)
ax_onb.set_ylabel("Temperature difference [K]")
ax_onb.set_title(
    "ONB Criterion Debug: DeltaT_actual vs DeltaT_ONB",
    fontsize=14,
    fontweight="bold"
)
ax_onb.grid(True, linestyle="--", alpha=0.5)
ax_onb.legend(loc="best")

plt.tight_layout()
plt.savefig(
    "steam_generator_onb_debug.png",
    dpi=300
)
plt.close(fig_onb)


# ============================================================
# Print results
# ============================================================

print("\n==============================")
print("Steam Generator Results")
print("==============================")

print("[Direction]")
print("Cold water : left  -> right")
print("Hot water  : right -> left")

print("\n[Inlet boundary conditions]")
print(f"Hot inlet pressure : {P_hot_in / 1e6:.2f} MPa")
print(f"Hot inlet temperature : {T_hot_in:.2f} K")
print(f"Hot inlet velocity : {u_hot_in:.2f} m/s")
print(f"Hot mass flow rate : {m_dot_hot:.5f} kg/s")
print(f"Hot mass flux : {G_hot:.2f} kg/m2-s")
print(f"Cold inlet pressure : {P_cold_in / 1e6:.2f} MPa")
print(f"Cold inlet temperature : {T_cold_in:.2f} K")
print(f"Cold inlet velocity : {u_cold_in:.2f} m/s")
print(f"Cold mass flow rate : {m_dot_cold:.5f} kg/s")
print(f"Cold mass flux : {G_cold:.2f} kg/m2-s")

print("\n[Temperature]")
print(f"Hot inlet  at x = L : {T_hot[-1] - 273.15:.2f} degC")
print(f"Hot outlet at x = 0 : {T_hot[0] - 273.15:.2f} degC")
print(f"Cold inlet  at x = 0 : {T_cold[0] - 273.15:.2f} degC")
print(f"Cold outlet at x = L : {T_cold[-1] - 273.15:.2f} degC")
print(f"Cold outlet target at x = L : {T_cold_out_target - 273.15:.2f} degC")
print(f"Max hot-side wall temperature : {np.max(T_wall_hot) - 273.15:.2f} degC")
print(f"Max cold-side wall temperature : {np.max(T_wall_cold) - 273.15:.2f} degC")
print(f"Hot pressure at x = 0 : {P_hot_node[0] / 1e6:.3f} MPa")
print(f"Hot pressure at x = L : {P_hot_node[-1] / 1e6:.3f} MPa")
print(f"Cold pressure at x = 0 : {P_cold_node[0] / 1e6:.3f} MPa")
print(f"Cold pressure at x = L : {P_cold_node[-1] / 1e6:.3f} MPa")

if T_cold[-1] + 0.1 < T_cold_out_target:
    print(
        "Warning: cold outlet temperature is below 290 degC; "
        "heat exchanger length or heat duty is insufficient."
    )

if x_onb is not None:
    print(f"ONB starts by Bergles-Rohsenow criterion at x = {x_onb:.3f} m")
else:
    print("ONB not detected by Bergles-Rohsenow criterion.")
    print("ONB not detected because DeltaT_actual < DeltaT_ONB at all nodes")

print("\n[ONB debug]")
print(f"max(DeltaT_actual) : {max_deltaT_actual:.6f} K")
print(f"min(DeltaT_ONB) : {min_deltaT_ONB:.6f} K")
print(f"max(ONB_margin) : {max_ONB_margin:.6f} K")
print(f"x at max(ONB_margin) : {x_max_ONB_margin:.6f} m")

if x_chf is not None:
    print(f"CHF starts by Del Col dryout criterion at x = {x_chf:.3f} m")
else:
    print("CHF not detected by Del Col dryout criterion.")

if x_superheated is not None:
    print(f"Superheated Steam starts at x = {x_superheated:.3f} m")
else:
    print("Superheated Steam not reached.")

print("\n[Cold-side phase change]")
print(f"T_sat at cold inlet pressure : {T_sat_node[0] - 273.15:.2f} degC")
print(f"T_sat at cold outlet pressure : {T_sat_node[-1] - 273.15:.2f} degC")
print(f"Latent heat H_fg at cold inlet : {H_fg / 1e3:.2f} kJ/kg")
print(f"Cold outlet quality at x = L : {quality[-1]:.3f}")
print(f"Cold outlet phase : {phase[-1]}")

print("\n[Pressure drop]")
print(f"Pressure drop model in boiling region : {pressure_drop_model}")
print(f"Total cold-side pressure drop : {dP_cold_cumulative[-1] / 1000.0:.3f} kPa")
print(f"Total hot-side pressure drop  : {np.sum(dP_hot_cell) / 1000.0:.3f} kPa")

finite_xdi_raw = x_di_raw_cell[np.isfinite(x_di_raw_cell)]
finite_xdi = x_di_cell[np.isfinite(x_di_cell)]
if finite_xdi.size > 0:
    print("\n[CHF / Dryout criterion]")
    print("Dryout x_di is calculated from Del Col et al. (2010) local conditions.")
    print(f"Del Col local x_di range : {np.min(finite_xdi):.3f} - {np.max(finite_xdi):.3f}")

print("\n[Cold-side heat transfer correlation]")
print(df_cell["cold_heat_transfer_correlation"].value_counts())

print("\n[Cold-side pressure drop correlation]")
print(df_cell["pressure_drop_correlation"].value_counts())

print("\nCSV saved:")
print(node_csv_file)
print(cell_csv_file)
print("Figure saved:") 
print("steam_generator_temperature_pressure_drop.png")
print("steam_generator_onb_debug.png")
print("==============================")
 
