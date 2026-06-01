# steam_generator_counterflow.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from datetime import datetime
from pathlib import Path
from CoolProp.CoolProp import PropsSI

FLIP_PLOT_LEFT_RIGHT = True


def save_csv_with_fallback(df, filename, **kwargs):
    path = Path(filename)
    try:
        df.to_csv(path, **kwargs)
        return path
    except PermissionError:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback_path = path.with_name(f"{path.stem}_{timestamp}{path.suffix}")
        df.to_csv(fallback_path, **kwargs)
        print(
            f"[Warning] Could not write {path} because it is open or locked. "
            f"Saved as {fallback_path} instead."
        )
        return fallback_path


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


def friction_factor_darcy(Re):
    """
    Darcy friction factor for internal flow.
    Laminar: f = 64 / Re
    Turbulent smooth pipe: Blasius correlation
    """

    Re = max(Re, 1.0)

    if Re < 2300.0:
        return 64.0 / Re

    return 0.3164 / Re**0.25


def friction_factor(Re):
    return friction_factor_darcy(Re)


def single_phase_dpdz(G, rho, mu, Dh):
    """
    Single-phase frictional pressure gradient [Pa/m].
    G is mass flux [kg/m2s].
    """

    G = max(G, 1e-12)
    rho = max(rho, 1e-12)
    mu = max(mu, 1e-12)
    Dh = max(Dh, 1e-12)

    Re = G * Dh / mu
    f = friction_factor_darcy(Re)
    dpdz = f / Dh * G**2 / (2.0 * rho)

    return dpdz


def single_phase_pressure_drop(f, dx, Dh, rho, velocity):
    """Darcy-Weisbach pressure drop for one cell."""

    return f * dx / Dh * 0.5 * rho * velocity**2


def get_saturation_props(P, fluid):
    T_sat = PropsSI("T", "P", P, "Q", 0, fluid)
    H_f = PropsSI("H", "P", P, "Q", 0, fluid)
    H_g = PropsSI("H", "P", P, "Q", 1, fluid)
    H_fg = H_g - H_f

    return T_sat, H_f, H_g, H_fg


def get_cold_state(Hc, P_cold, fluid_cold):

    T_sat, H_f, H_g, H_fg = get_saturation_props(P_cold, fluid_cold)

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

    return T, quality, phase, rho, mu, k, cp, T_sat, H_f, H_g, H_fg


def describe_cold_temperature_change(H_in, H_out, P_cold, fluid_cold):
    """
    Classify how cold-side temperature changes across one cell.

    Single-phase cells change temperature with sensible heat.
    Two-phase cells stay near saturation temperature while quality changes.
    """

    _T_sat, H_f, H_g, _H_fg = get_saturation_props(P_cold, fluid_cold)

    if H_out <= H_f:
        return "single_phase_liquid_sensible"

    if H_in >= H_g:
        return "single_phase_vapor_sensible"

    if H_in < H_f and H_out <= H_g:
        return "liquid_to_two_phase"

    if H_f <= H_in <= H_g and H_out <= H_g:
        return "two_phase_quality_change"

    if H_f <= H_in <= H_g and H_out > H_g:
        return "two_phase_to_superheated"

    if H_in < H_f and H_out > H_g:
        return "liquid_to_superheated"

    return "single_phase_sensible"


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
    Separated-flow two-phase frictional pressure gradient [Pa/m].

    (dP/dz)_TP = (dP/dz)_f * phi_f^2
    Default model: Mishima & Hibiki (1996).
    """

    x = np.clip(quality, 1e-6, 0.999999)
    G_f = max(G * (1.0 - x), 1e-12)
    G_g = max(G * x, 1e-12)

    Re_f = G_f * Dh / max(mu_l, 1e-12)
    Re_g = G_g * Dh / max(mu_g, 1e-12)

    dpdz_f = single_phase_dpdz(G_f, rho_l, mu_l, Dh)
    dpdz_g = single_phase_dpdz(G_g, rho_g, mu_g, Dh)
    X = np.sqrt(max(dpdz_f, 1e-30) / max(dpdz_g, 1e-30))

    model_key = model.strip().lower()
    shape_key = channel_shape.strip().lower()

    if model_key in ("mishima-hibiki", "mishima & hibiki", "mishima_hibiki"):
        if shape_key == "rectangular":
            C = 21.0 * (1.0 - np.exp(-0.319 * Dh))
        else:
            C = 21.0 * (1.0 - np.exp(-0.333 * Dh))

        phi_f2 = 1.0 + C / X + 1.0 / X**2

    elif model_key in ("yu", "yu et al.", "yu et al. 2002", "yu-2002"):
        bracket = (
            18.65
            * (rho_g / rho_l)**0.5
            * ((1.0 - x) / x)
            * (Re_g**0.1 / max(Re_f, 1e-30)**0.5)
        )
        phi_f2 = max(bracket, 1e-30)**(-1.9)

    elif model_key in ("sun-mishima", "sun & mishima", "sun_mishima"):
        # The requested function signature does not include surface tension,
        # so N_conf is kept as a configurable local assumption.
        N_conf = 1.0

        if Re_f < 2000.0 and Re_g < 2000.0:
            C = (
                26.0
                * (1.0 + Re_f / 1000.0)
                * (1.0 - np.exp(-0.153 / (0.27 * N_conf + 0.8)))
            )
            phi_f2 = 1.0 + C / X + 1.0 / X**2
        else:
            C = 1.79 * (Re_g / max(Re_f, 1e-30))**0.4 * ((1.0 - x) / x)**0.5
            phi_f2 = 1.0 + C / X**1.19 + 1.0 / X**2

    else:
        raise ValueError(
            "Unknown two-phase pressure drop model. "
            "Use 'Mishima-Hibiki', 'Yu', or 'Sun-Mishima'."
        )

    dpdz_tp = dpdz_f * phi_f2

    return dpdz_tp, phi_f2, X, Re_f, Re_g


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

P_hot_in = 15e6
T_hot_in_C = 300.0
T_hot_in = T_hot_in_C + 273.15
V_hot_in = 2.0
rho_hot_in = PropsSI("D", "P", P_hot_in, "T", T_hot_in, fluid_hot)
m_dot_hot = rho_hot_in * A_flow * V_hot_in


# ============================================================
# Cold side : RIGHT -> LEFT
# ============================================================

fluid_cold = "Water"

m_dot_cold = 0.1
P_cold_in = 6e6

T_cold_in_C = 280.0
T_cold_in = T_cold_in_C + 273.15

cold_two_phase_dp_model = "Mishima-Hibiki"
cold_channel_shape = "circular"


def calculate_case(m_dot_cold_case, P_cold_in_case, T_cold_in_C_case):
    T_hot_in_case = T_hot_in_C + 273.15
    rho_hot_in_case = PropsSI("D", "P", P_hot_in, "T", T_hot_in_case, fluid_hot)
    m_dot_hot_case = rho_hot_in_case * A_flow * V_hot_in

    T_cold_in_case = T_cold_in_C_case + 273.15

    T_hot_case = np.zeros(N + 1)
    H_hot_case = np.zeros(N + 1)
    P_hot_case = np.zeros(N + 1)

    T_cold_case = np.zeros(N + 1)
    H_cold_case = np.zeros(N + 1)
    P_cold_case = np.zeros(N + 1)

    quality_case = np.zeros(N + 1)
    phase_case = [""] * (N + 1)

    h_hot_case = np.zeros(N)
    h_cold_case = np.zeros(N)
    U_case = np.zeros(N)
    q_case = np.zeros(N)
    q_flux_case = np.zeros(N)
    cold_correlation_case = [""] * N
    cold_temperature_change_case = [""] * N

    dP_hot_case = np.zeros(N)
    dP_cold_case = np.zeros(N)
    dpdz_cold_case = np.zeros(N)
    cold_cumulative_dP_case = np.zeros(N + 1)
    f_hot_case = np.zeros(N)
    f_cold_case = np.zeros(N)
    Re_hot_case = np.zeros(N)
    Re_cold_case = np.zeros(N)
    Re_cold_liquid_case = np.zeros(N)
    Re_cold_vapor_case = np.zeros(N)
    phi_f2_cold_case = np.ones(N)
    X_cold_case = np.zeros(N)
    T_sat_cold_case = np.zeros(N + 1)
    H_f_cold_case = np.zeros(N + 1)
    H_g_cold_case = np.zeros(N + 1)
    H_fg_cold_case = np.zeros(N + 1)

    T_hot_case[0] = T_hot_in_case
    P_hot_case[0] = P_hot_in
    H_hot_case[0] = PropsSI("H", "P", P_hot_case[0], "T", T_hot_in_case, fluid_hot)

    T_cold_case[N] = T_cold_in_case
    P_cold_case[N] = P_cold_in_case
    H_cold_case[N] = PropsSI("H", "P", P_cold_case[N], "T", T_cold_in_case, fluid_cold)

    G_cold_case = m_dot_cold_case / A_flow

    for i in range(N):
        ih = i
        ic = N - i

        P_hot_local = P_hot_case[ih]
        P_cold_local = P_cold_case[ic]

        T_hot_case[ih] = PropsSI("T", "P", P_hot_local, "H", H_hot_case[ih], fluid_hot)
        rho_h = PropsSI("D", "P", P_hot_local, "H", H_hot_case[ih], fluid_hot)
        mu_h = PropsSI("V", "P", P_hot_local, "H", H_hot_case[ih], fluid_hot)
        k_h = PropsSI("L", "P", P_hot_local, "H", H_hot_case[ih], fluid_hot)
        cp_h = PropsSI("C", "P", P_hot_local, "H", H_hot_case[ih], fluid_hot)

        V_h = m_dot_hot_case / (rho_h * A_flow)
        Re_h = rho_h * V_h * D_inner / mu_h
        Pr_h = cp_h * mu_h / k_h
        f_hot_local = friction_factor(Re_h)

        h_hot_local = dittus_boelter(
            Re=Re_h,
            Pr=Pr_h,
            k=k_h,
            Dh=D_inner,
            heating=False
        )

        (
            T_cold_case[ic],
            quality_case[ic],
            phase_case[ic],
            rho_c,
            mu_c,
            k_c,
            cp_c,
            T_sat_cold_case[ic],
            H_f_cold_case[ic],
            H_g_cold_case[ic],
            H_fg_cold_case[ic]
        ) = get_cold_state(H_cold_case[ic], P_cold_local, fluid_cold)

        V_c = m_dot_cold_case / (rho_c * A_flow)
        Re_c = rho_c * V_c * D_inner / mu_c
        Pr_c = cp_c * mu_c / k_c
        f_cold_local = friction_factor(Re_c)

        h_cold_local, corr_name = get_cold_heat_transfer_coefficient(
            phase=phase_case[ic],
            Re_c=Re_c,
            Pr_c=Pr_c,
            k_c=k_c,
            D_inner=D_inner,
            q_flux=50000.0,
            G_cold=G_cold_case,
            H_fg=H_fg_cold_case[ic],
            quality_local=quality_case[ic]
        )

        U_local = 1.0 / (1.0 / h_hot_local + t_wall / k_wall + 1.0 / h_cold_local)
        dT = T_hot_case[ih] - T_cold_case[ic]

        if dT <= 0.0:
            q_local = 0.0
            q_flux_local = 0.0
        else:
            q_local = U_local * P_heat * dx * dT
            q_flux_local = q_local / (P_heat * dx)
            H_cold_out_guess = H_cold_case[ic] + q_local / m_dot_cold_case
            cold_temperature_change_guess = describe_cold_temperature_change(
                H_in=H_cold_case[ic],
                H_out=H_cold_out_guess,
                P_cold=P_cold_local,
                fluid_cold=fluid_cold
            )
            needs_two_phase_recalc = (
                phase_case[ic] == "boiling"
                or "two_phase" in cold_temperature_change_guess
                or cold_temperature_change_guess == "liquid_to_superheated"
            )

            if needs_two_phase_recalc:
                H_cold_mid = 0.5 * (H_cold_case[ic] + H_cold_out_guess)
                (
                    _T_cold_mid,
                    quality_mid,
                    phase_mid,
                    rho_c_mid,
                    mu_c_mid,
                    k_c_mid,
                    cp_c_mid,
                    _T_sat_mid,
                    _H_f_mid,
                    _H_g_mid,
                    H_fg_mid
                ) = get_cold_state(H_cold_mid, P_cold_local, fluid_cold)

                V_c_mid = m_dot_cold_case / (rho_c_mid * A_flow)
                Re_c_mid = rho_c_mid * V_c_mid * D_inner / mu_c_mid
                Pr_c_mid = cp_c_mid * mu_c_mid / k_c_mid
                phase_for_h = (
                    "boiling"
                    if (
                        "two_phase" in cold_temperature_change_guess
                        or cold_temperature_change_guess == "liquid_to_superheated"
                    )
                    else phase_mid
                )

                h_cold_local, corr_name = get_cold_heat_transfer_coefficient(
                    phase=phase_for_h,
                    Re_c=Re_c_mid,
                    Pr_c=Pr_c_mid,
                    k_c=k_c_mid,
                    D_inner=D_inner,
                    q_flux=q_flux_local,
                    G_cold=G_cold_case,
                    H_fg=H_fg_mid,
                    quality_local=quality_mid
                )

                U_local = 1.0 / (
                    1.0 / h_hot_local
                    + t_wall / k_wall
                    + 1.0 / h_cold_local
                )
                q_local = U_local * P_heat * dx * dT
                q_flux_local = q_local / (P_heat * dx)

        h_hot_case[i] = h_hot_local
        h_cold_case[i] = h_cold_local
        U_case[i] = U_local
        q_case[i] = q_local
        q_flux_case[i] = q_flux_local
        cold_correlation_case[i] = corr_name
        f_hot_case[i] = f_hot_local
        f_cold_case[i] = f_cold_local
        Re_hot_case[i] = Re_h
        Re_cold_case[i] = Re_c

        H_hot_case[ih + 1] = H_hot_case[ih] - q_local / m_dot_hot_case
        dP_hot_local = single_phase_pressure_drop(
            f=f_hot_local,
            dx=dx,
            Dh=D_inner,
            rho=rho_h,
            velocity=V_h
        )
        dP_hot_case[i] = dP_hot_local
        P_hot_case[ih + 1] = max(P_hot_case[ih] - dP_hot_local, 1.0e5)

        H_cold_in_cell = H_cold_case[ic]
        H_cold_case[ic - 1] = H_cold_in_cell + q_local / m_dot_cold_case
        cold_temperature_change_case[i] = describe_cold_temperature_change(
            H_in=H_cold_in_cell,
            H_out=H_cold_case[ic - 1],
            P_cold=P_cold_local,
            fluid_cold=fluid_cold
        )

        cold_cell_has_two_phase = (
            phase_case[ic] == "boiling"
            or "two_phase" in cold_temperature_change_case[i]
            or cold_temperature_change_case[i] == "liquid_to_superheated"
        )

        if cold_cell_has_two_phase:
            H_cold_dp = 0.5 * (H_cold_in_cell + H_cold_case[ic - 1])
            (
                _T_cold_dp,
                quality_dp,
                _phase_dp,
                _rho_c_dp,
                _mu_c_dp,
                _k_c_dp,
                _cp_c_dp,
                _T_sat_dp,
                _H_f_dp,
                _H_g_dp,
                _H_fg_dp
            ) = get_cold_state(H_cold_dp, P_cold_local, fluid_cold)

            rho_l = PropsSI("D", "P", P_cold_local, "Q", 0, fluid_cold)
            rho_g = PropsSI("D", "P", P_cold_local, "Q", 1, fluid_cold)
            mu_l = PropsSI("V", "P", P_cold_local, "Q", 0, fluid_cold)
            mu_g = PropsSI("V", "P", P_cold_local, "Q", 1, fluid_cold)

            dpdz_cold_local, phi_f2, X_tt, Re_f, Re_g = separated_flow_pressure_drop(
                quality=quality_dp,
                G=G_cold_case,
                Dh=D_inner,
                rho_l=rho_l,
                rho_g=rho_g,
                mu_l=mu_l,
                mu_g=mu_g,
                model=cold_two_phase_dp_model,
                channel_shape=cold_channel_shape
            )
            dP_cold_local = dpdz_cold_local * dx
            Re_cold_liquid_case[i] = Re_f
            Re_cold_vapor_case[i] = Re_g
            phi_f2_cold_case[i] = phi_f2
            X_cold_case[i] = X_tt
        else:
            dpdz_cold_local = single_phase_dpdz(
                G=G_cold_case,
                rho=rho_c,
                mu=mu_c,
                Dh=D_inner
            )
            dP_cold_local = dpdz_cold_local * dx
            Re_cold_liquid_case[i] = Re_c
            Re_cold_vapor_case[i] = 0.0
            phi_f2_cold_case[i] = 1.0
            X_cold_case[i] = 0.0

        dpdz_cold_case[i] = dpdz_cold_local
        dP_cold_case[i] = dP_cold_local
        cold_cumulative_dP_case[ic - 1] = cold_cumulative_dP_case[ic] + dP_cold_local / 1e3
        P_cold_case[ic - 1] = max(P_cold_case[ic] - dP_cold_local, 1.0e5)

    for i in range(N + 1):
        T_hot_case[i] = PropsSI("T", "P", P_hot_case[i], "H", H_hot_case[i], fluid_hot)
        (
            T_cold_case[i],
            quality_case[i],
            phase_case[i],
            _rho_c,
            _mu_c,
            _k_c,
            _cp_c,
            T_sat_cold_case[i],
            H_f_cold_case[i],
            H_g_cold_case[i],
            H_fg_cold_case[i]
        ) = get_cold_state(H_cold_case[i], P_cold_case[i], fluid_cold)

    phase_arr_case = np.array(phase_case)
    T_cold_plot_C_case = (T_cold_case - 273.15).copy()
    boiling_indices_case = np.where(phase_arr_case == "boiling")[0]

    if boiling_indices_case.size > 0:
        T_cold_plot_C_case[boiling_indices_case] = T_cold_plot_C_case[boiling_indices_case[-1]]

    hot_cumulative_dP_case = (P_hot_case[0] - P_hot_case) / 1e3
    cold_cumulative_dP_from_inlet_case = (P_cold_case[-1] - P_cold_case) / 1e3
    cold_temperature_change_counts_case = pd.Series(cold_temperature_change_case).value_counts()
    cold_temperature_change_text_case = "\n".join(
        f"- {mode}: {count}"
        for mode, count in cold_temperature_change_counts_case.items()
    )

    condition_text_case = (
        "Input conditions\n"
        f"- Total length: {L:.2f} m\n"
        f"- Nodes: {N:d}\n"
        f"- Node length: {dx:.3f} m\n"
        "\n"
        "Hot side\n"
        f"- Inlet temperature: {T_hot_in_C:.1f} degC\n"
        f"- Inlet pressure: {P_hot_in / 1e6:.2f} MPa\n"
        f"- Inlet velocity: {V_hot_in:.2f} m/s\n"
        f"- Mass flow rate: {m_dot_hot_case:.4f} kg/s\n"
        f"- Direction: x = 0 -> L\n"
        "\n"
        "Cold side\n"
        f"- Inlet temperature: {T_cold_in_C_case:.1f} degC\n"
        f"- Inlet pressure: {P_cold_in_case / 1e6:.2f} MPa\n"
        f"- Mass flow rate: {m_dot_cold_case:.4f} kg/s\n"
        f"- Direction: x = L -> 0\n"
        "\n"
        "Results\n"
        f"- Hot outlet temperature: {T_hot_case[-1] - 273.15:.2f} degC\n"
        f"- Cold outlet temperature: {T_cold_case[0] - 273.15:.2f} degC\n"
        f"- Hot pressure drop: {hot_cumulative_dP_case[-1]:.3f} kPa\n"
        f"- Cold pressure drop: {cold_cumulative_dP_from_inlet_case[0]:.3f} kPa\n"
        "\n"
        "Cold-side temperature mode\n"
        f"{cold_temperature_change_text_case}"
    )

    return {
        "x": x,
        "T_hot_C": T_hot_case - 273.15,
        "T_cold_C": T_cold_case - 273.15,
        "T_cold_plot_C": T_cold_plot_C_case,
        "T_sat_cold_C": T_sat_cold_case - 273.15,
        "hot_cumulative_dP": hot_cumulative_dP_case,
        "cold_cumulative_dP_from_inlet": cold_cumulative_dP_from_inlet_case,
        "phase_arr": phase_arr_case,
        "boiling_indices": boiling_indices_case,
        "superheated_indices": np.where(phase_arr_case == "superheated_vapor")[0],
        "condition_text": condition_text_case,
    }


# ============================================================
# Arrays
# ============================================================

x = np.linspace(0.0, L, N + 1)

T_hot = np.zeros(N + 1)
H_hot = np.zeros(N + 1)
P_hot = np.zeros(N + 1)

T_cold = np.zeros(N + 1)
H_cold = np.zeros(N + 1)
P_cold = np.zeros(N + 1)

quality = np.zeros(N + 1)
phase = [""] * (N + 1)

h_hot_arr = np.zeros(N)
h_cold_arr = np.zeros(N)
U_arr = np.zeros(N)
q_arr = np.zeros(N)
q_flux_arr = np.zeros(N)
cold_correlation = [""] * N
cold_temperature_change = [""] * N

dP_hot_arr = np.zeros(N)
dP_cold_arr = np.zeros(N)
dpdz_cold_arr = np.zeros(N)
cold_cumulative_dP = np.zeros(N + 1)
f_hot_arr = np.zeros(N)
f_cold_arr = np.zeros(N)
Re_hot_arr = np.zeros(N)
Re_cold_arr = np.zeros(N)
Re_cold_liquid_arr = np.zeros(N)
Re_cold_vapor_arr = np.zeros(N)
phi_f2_cold_arr = np.ones(N)
X_cold_arr = np.zeros(N)
T_sat_cold = np.zeros(N + 1)
H_f_cold = np.zeros(N + 1)
H_g_cold = np.zeros(N + 1)
H_fg_cold = np.zeros(N + 1)


# ============================================================
# Initial condition
# ============================================================

# Hot inlet: x = 0, left side
T_hot[0] = T_hot_in
P_hot[0] = P_hot_in
H_hot[0] = PropsSI(
    "H",
    "P", P_hot[0],
    "T", T_hot_in,
    fluid_hot
)

# Cold inlet: x = L, right side
T_cold[N] = T_cold_in
P_cold[N] = P_cold_in
H_cold[N] = PropsSI(
    "H",
    "P", P_cold[N],
    "T", T_cold_in,
    fluid_cold
)


# ============================================================
# Cold-side saturation properties
# ============================================================

T_sat_in = PropsSI("T", "P", P_cold_in, "Q", 0, fluid_cold)

H_f_in = PropsSI("H", "P", P_cold_in, "Q", 0, fluid_cold)
H_g_in = PropsSI("H", "P", P_cold_in, "Q", 1, fluid_cold)

H_fg_in = H_g_in - H_f_in

if T_cold_in >= T_sat_in:
    print(
        "[Warning] Cold inlet is not subcooled liquid at this pressure. "
        f"At P_cold_in = {P_cold_in / 1e6:.2f} MPa, "
        f"T_sat = {T_sat_in - 273.15:.2f} °C."
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

    P_hot_local = P_hot[ih]
    P_cold_local = P_cold[ic]

    # =====================================================
    # Hot side properties
    # =====================================================

    T_hot[ih] = PropsSI(
        "T",
        "P", P_hot_local,
        "H", H_hot[ih],
        fluid_hot
    )

    rho_h = PropsSI("D", "P", P_hot_local, "H", H_hot[ih], fluid_hot)
    mu_h = PropsSI("V", "P", P_hot_local, "H", H_hot[ih], fluid_hot)
    k_h = PropsSI("L", "P", P_hot_local, "H", H_hot[ih], fluid_hot)
    cp_h = PropsSI("C", "P", P_hot_local, "H", H_hot[ih], fluid_hot)

    V_h = m_dot_hot / (rho_h * A_flow)
    Re_h = rho_h * V_h * D_inner / mu_h
    Pr_h = cp_h * mu_h / k_h
    f_hot = friction_factor(Re_h)

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
        cp_c,
        T_sat_cold[ic],
        H_f_cold[ic],
        H_g_cold[ic],
        H_fg_cold[ic]
    ) = get_cold_state(
        H_cold[ic],
        P_cold_local,
        fluid_cold
    )

    V_c = m_dot_cold / (rho_c * A_flow)
    Re_c = rho_c * V_c * D_inner / mu_c
    Pr_c = cp_c * mu_c / k_c
    f_cold = friction_factor(Re_c)

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
        H_fg=H_fg_cold[ic],
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

        H_cold_out_guess = H_cold[ic] + q / m_dot_cold
        cold_temperature_change_guess = describe_cold_temperature_change(
            H_in=H_cold[ic],
            H_out=H_cold_out_guess,
            P_cold=P_cold_local,
            fluid_cold=fluid_cold
        )

        needs_two_phase_recalc = (
            phase[ic] == "boiling"
            or "two_phase" in cold_temperature_change_guess
            or cold_temperature_change_guess == "liquid_to_superheated"
        )

        # If the cell is boiling or crosses into/out of the two-phase region,
        # recalculate cold-side h from a representative mid-cell state.
        if needs_two_phase_recalc:

            H_cold_mid = 0.5 * (H_cold[ic] + H_cold_out_guess)

            (
                T_cold_mid,
                quality_mid,
                phase_mid,
                rho_c_mid,
                mu_c_mid,
                k_c_mid,
                cp_c_mid,
                T_sat_mid,
                H_f_mid,
                H_g_mid,
                H_fg_mid
            ) = get_cold_state(
                H_cold_mid,
                P_cold_local,
                fluid_cold
            )

            V_c_mid = m_dot_cold / (rho_c_mid * A_flow)
            Re_c_mid = rho_c_mid * V_c_mid * D_inner / mu_c_mid
            Pr_c_mid = cp_c_mid * mu_c_mid / k_c_mid
            phase_for_h = (
                "boiling"
                if (
                    "two_phase" in cold_temperature_change_guess
                    or cold_temperature_change_guess == "liquid_to_superheated"
                )
                else phase_mid
            )

            h_cold, corr_name = get_cold_heat_transfer_coefficient(
                phase=phase_for_h,
                Re_c=Re_c_mid,
                Pr_c=Pr_c_mid,
                k_c=k_c_mid,
                D_inner=D_inner,
                q_flux=q_flux_local,
                G_cold=G_cold,
                H_fg=H_fg_mid,
                quality_local=quality_mid
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
    f_hot_arr[i] = f_hot
    f_cold_arr[i] = f_cold
    Re_hot_arr[i] = Re_h
    Re_cold_arr[i] = Re_c

    # =====================================================
    # Update hot side
    # Hot water loses heat
    # x = 0 -> L
    # =====================================================

    H_hot[ih + 1] = H_hot[ih] - q / m_dot_hot

    dP_hot = single_phase_pressure_drop(
        f=f_hot,
        dx=dx,
        Dh=D_inner,
        rho=rho_h,
        velocity=V_h
    )
    dP_hot_arr[i] = dP_hot
    P_hot[ih + 1] = max(P_hot[ih] - dP_hot, 1.0e5)

    # =====================================================
    # Update cold side
    # Cold water gains heat
    # x = L -> 0
    # =====================================================

    H_cold_in_cell = H_cold[ic]
    H_cold[ic - 1] = H_cold_in_cell + q / m_dot_cold
    cold_temperature_change[i] = describe_cold_temperature_change(
        H_in=H_cold_in_cell,
        H_out=H_cold[ic - 1],
        P_cold=P_cold_local,
        fluid_cold=fluid_cold
    )

    cold_cell_has_two_phase = (
        phase[ic] == "boiling"
        or "two_phase" in cold_temperature_change[i]
        or cold_temperature_change[i] == "liquid_to_superheated"
    )

    if cold_cell_has_two_phase:
        H_cold_dp = 0.5 * (H_cold_in_cell + H_cold[ic - 1])
        (
            T_cold_dp,
            quality_dp,
            phase_dp,
            rho_c_dp,
            mu_c_dp,
            k_c_dp,
            cp_c_dp,
            T_sat_dp,
            H_f_dp,
            H_g_dp,
            H_fg_dp
        ) = get_cold_state(
            H_cold_dp,
            P_cold_local,
            fluid_cold
        )

        rho_l = PropsSI("D", "P", P_cold_local, "Q", 0, fluid_cold)
        rho_g = PropsSI("D", "P", P_cold_local, "Q", 1, fluid_cold)
        mu_l = PropsSI("V", "P", P_cold_local, "Q", 0, fluid_cold)
        mu_g = PropsSI("V", "P", P_cold_local, "Q", 1, fluid_cold)

        dpdz_cold, phi_f2, X_tt, Re_f, Re_g = separated_flow_pressure_drop(
            quality=quality_dp,
            G=G_cold,
            Dh=D_inner,
            rho_l=rho_l,
            rho_g=rho_g,
            mu_l=mu_l,
            mu_g=mu_g,
            model=cold_two_phase_dp_model,
            channel_shape=cold_channel_shape
        )

        dP_cold = dpdz_cold * dx
        Re_cold_liquid_arr[i] = Re_f
        Re_cold_vapor_arr[i] = Re_g
        phi_f2_cold_arr[i] = phi_f2
        X_cold_arr[i] = X_tt
    else:
        dpdz_cold = single_phase_dpdz(
            G=G_cold,
            rho=rho_c,
            mu=mu_c,
            Dh=D_inner
        )
        dP_cold = dpdz_cold * dx
        Re_cold_liquid_arr[i] = Re_c
        Re_cold_vapor_arr[i] = 0.0
        phi_f2_cold_arr[i] = 1.0
        X_cold_arr[i] = 0.0

    dpdz_cold_arr[i] = dpdz_cold
    dP_cold_arr[i] = dP_cold
    cold_cumulative_dP[ic - 1] = cold_cumulative_dP[ic] + dP_cold / 1e3
    P_cold[ic - 1] = max(P_cold[ic] - dP_cold, 1.0e5)


# ============================================================
# Final state conversion
# ============================================================

for i in range(N + 1):

    T_hot[i] = PropsSI(
        "T",
        "P", P_hot[i],
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
        cp_c,
        T_sat_cold[i],
        H_f_cold[i],
        H_g_cold[i],
        H_fg_cold[i]
    ) = get_cold_state(
        H_cold[i],
        P_cold[i],
        fluid_cold
    )


# ============================================================
# Dataframe
# ============================================================

df_node = pd.DataFrame({
    "x_m": x,
    "P_hot_MPa": P_hot / 1e6,
    "P_cold_MPa": P_cold / 1e6,
    "cold_cumulative_dP_kPa": cold_cumulative_dP,
    "T_hot_C": T_hot - 273.15,
    "T_cold_C": T_cold - 273.15,
    "T_sat_cold_C": T_sat_cold - 273.15,
    "cold_superheat_C": np.maximum(T_cold - T_sat_cold, 0.0),
    "H_hot_Jkg": H_hot,
    "H_cold_Jkg": H_cold,
    "H_f_cold_Jkg": H_f_cold,
    "H_g_cold_Jkg": H_g_cold,
    "H_fg_cold_Jkg": H_fg_cold,
    "quality": quality,
    "cold_phase": phase
})

df_cell = pd.DataFrame({
    "cell": np.arange(N),
    "x_hot_mid_m": 0.5 * (x[:-1] + x[1:]),
    "x_cold_mid_m": 0.5 * (x[N - np.arange(N)] + x[N - np.arange(N) - 1]),
    "h_hot_W_m2K": h_hot_arr,
    "h_cold_W_m2K": h_cold_arr,
    "U_W_m2K": U_arr,
    "q_W": q_arr,
    "q_flux_W_m2": q_flux_arr,
    "dP_hot_Pa": dP_hot_arr,
    "dpdz_cold_Pa_m": dpdz_cold_arr,
    "dP_cold_cell_Pa": dP_cold_arr,
    "cold_cumulative_dP_kPa": cold_cumulative_dP[N - np.arange(N) - 1],
    "f_hot": f_hot_arr,
    "f_cold": f_cold_arr,
    "Re_hot": Re_hot_arr,
    "Re_cold": Re_cold_arr,
    "Re_cold_liquid": Re_cold_liquid_arr,
    "Re_cold_vapor": Re_cold_vapor_arr,
    "phi_f2_cold": phi_f2_cold_arr,
    "X_cold": X_cold_arr,
    "cold_two_phase_dp_model": cold_two_phase_dp_model,
    "cold_correlation": cold_correlation,
    "cold_temperature_change": cold_temperature_change
})

node_csv_path = save_csv_with_fallback(
    df_node,
    "steam_generator_node_results.csv",
    index=False,
    encoding="utf-8-sig"
)

cell_csv_path = save_csv_with_fallback(
    df_cell,
    "steam_generator_cell_results.csv",
    index=False,
    encoding="utf-8-sig"
)

print(df_node)
print(df_cell)


# ============================================================
# Plot : one-page summary
# ============================================================

hot_cumulative_dP = (P_hot[0] - P_hot) / 1e3
cold_cumulative_dP_from_inlet = (P_cold[-1] - P_cold) / 1e3
cold_temperature_change_counts = pd.Series(cold_temperature_change).value_counts()
cold_temperature_change_text = "\n".join(
    f"- {mode}: {count}"
    for mode, count in cold_temperature_change_counts.items()
)

fig_temp = plt.figure(figsize=(14, 8))
grid = fig_temp.add_gridspec(
    2,
    2,
    width_ratios=[1.05, 2.1],
    height_ratios=[1.0, 1.0],
    wspace=0.28,
    hspace=0.32
)

ax_info = fig_temp.add_subplot(grid[:, 0])
ax_temp = fig_temp.add_subplot(grid[0, 1])
ax_pressure = fig_temp.add_subplot(grid[1, 1])

ax_info.axis("off")
condition_text = (
    "Input conditions\n"
    f"- Total length: {L:.2f} m\n"
    f"- Nodes: {N:d}\n"
    f"- Node length: {dx:.3f} m\n"
    "\n"
    "Hot side\n"
    f"- Inlet temperature: {T_hot_in_C:.1f} degC\n"
    f"- Inlet pressure: {P_hot_in / 1e6:.2f} MPa\n"
    f"- Inlet velocity: {V_hot_in:.2f} m/s\n"
    f"- Mass flow rate: {m_dot_hot:.4f} kg/s\n"
    f"- Direction: x = 0 -> L\n"
    "\n"
    "Cold side\n"
    f"- Inlet temperature: {T_cold_in_C:.1f} degC\n"
    f"- Inlet pressure: {P_cold_in / 1e6:.2f} MPa\n"
    f"- Mass flow rate: {m_dot_cold:.4f} kg/s\n"
    f"- Direction: x = L -> 0\n"
    "\n"
    "Results\n"
    f"- Hot outlet temperature: {T_hot[-1] - 273.15:.2f} degC\n"
    f"- Cold outlet temperature: {T_cold[0] - 273.15:.2f} degC\n"
    f"- Hot pressure drop: {hot_cumulative_dP[-1]:.3f} kPa\n"
    f"- Cold pressure drop: {cold_cumulative_dP_from_inlet[0]:.3f} kPa\n"
    "\n"
    "Cold-side temperature mode\n"
    f"{cold_temperature_change_text}"
)
info_text = ax_info.text(
    0.02,
    0.98,
    condition_text,
    va="top",
    ha="left",
    fontsize=11,
    linespacing=1.35,
    family="monospace"
)

phase_arr = np.array(phase)
T_cold_plot_C = (T_cold - 273.15).copy()
boiling_indices = np.where(phase_arr == "boiling")[0]

if boiling_indices.size > 0:
    boiling_temperature_C = T_cold_plot_C[boiling_indices[-1]]
    T_cold_plot_C[boiling_indices] = boiling_temperature_C

line_hot, = ax_temp.plot(
    x,
    T_hot - 273.15,
    color="red",
    linewidth=3,
    marker="o",
    markersize=3,
    label="Hot water: left → right"
)

line_cold, = ax_temp.plot(
    x,
    T_cold_plot_C,
    color="blue",
    linewidth=3,
    marker="o",
    markersize=4,
    drawstyle="steps-post",
    label="Cold water / steam: right → left"
)

line_sat, = ax_temp.plot(
    x,
    T_sat_cold - 273.15,
    color="black",
    linestyle="--",
    linewidth=1.5,
    label="Cold-side saturation temperature"
)

region_patches = []

if boiling_indices.size > 0:
    region_patches.append(
        ax_temp.axvspan(
            x[boiling_indices[0]],
            x[boiling_indices[-1]],
            color="skyblue",
            alpha=0.18,
            label="Boiling region: Shah correlation"
        )
    )

superheated_indices = np.where(phase_arr == "superheated_vapor")[0]

if superheated_indices.size > 0:
    region_patches.append(
        ax_temp.axvspan(
            x[superheated_indices[0]],
            x[superheated_indices[-1]],
            color="orange",
            alpha=0.14,
            label="Superheated vapor region"
        )
    )

ax_temp.text(
    0.05 * L,
    T_hot[0] - 273.15 + 5,
    "Hot inlet",
    color="red"
)

ax_temp.text(
    0.78 * L,
    T_hot[-1] - 273.15 + 5,
    "Hot outlet",
    color="red"
)

ax_temp.text(
    0.75 * L,
    T_cold[-1] - 273.15 - 25,
    "Cold inlet",
    color="blue"
)

ax_temp.text(
    0.03 * L,
    T_cold[0] - 273.15 + 5,
    "Cold outlet",
    color="blue"
)

ax_temp.annotate(
    "Hot flow",
    xy=(0.75 * L, min(T_hot - 273.15) + 20),
    xytext=(0.25 * L, min(T_hot - 273.15) + 20),
    arrowprops=dict(arrowstyle="->", color="red", lw=2),
    color="red",
    fontsize=11
)

ax_temp.annotate(
    "Cold flow",
    xy=(0.25 * L, min(T_cold - 273.15) - 10),
    xytext=(0.75 * L, min(T_cold - 273.15) - 10),
    arrowprops=dict(arrowstyle="->", color="blue", lw=2),
    color="blue",
    fontsize=11
)

if FLIP_PLOT_LEFT_RIGHT:
    ax_temp.invert_xaxis()
    xlabel = "Position x [m]  (left = L, right = 0)"
else:
    xlabel = "Position x [m]  (left = 0, right = L)"

ax_temp.set_xlabel(xlabel, fontsize=12)
ax_temp.set_ylabel("Temperature [°C]", fontsize=12)

ax_temp.set_title(
    "Temperature Change",
    fontsize=13,
    fontweight="bold"
)

ax_temp.grid(True, linestyle="--", alpha=0.5)
lines = [line_hot, line_cold, line_sat]
labels = [line.get_label() for line in lines]
ax_temp.legend(lines, labels, loc="best")
plt.tight_layout()


# ============================================================
# Plot : pressure
# ============================================================

line_hot_pressure, = ax_pressure.plot(
    x,
    hot_cumulative_dP,
    color="red",
    linewidth=3,
    marker="o",
    markersize=3,
    label="Hot-side cumulative pressure drop"
)

line_cold_pressure, = ax_pressure.plot(
    x,
    cold_cumulative_dP_from_inlet,
    color="blue",
    linewidth=3,
    marker="s",
    markersize=3,
    label="Cold-side cumulative pressure drop"
)

if FLIP_PLOT_LEFT_RIGHT:
    ax_pressure.invert_xaxis()

ax_pressure.set_xlabel(xlabel, fontsize=12)
ax_pressure.set_ylabel("Cumulative pressure drop [kPa]", fontsize=12)

ax_pressure.set_title(
    "Pressure Drop",
    fontsize=13,
    fontweight="bold"
)

ax_pressure.grid(True, linestyle="--", alpha=0.5)
ax_pressure.legend(loc="best")
plt.tight_layout()

fig_temp.suptitle(
    "Counter-flow Steam Generator Summary",
    fontsize=16,
    fontweight="bold"
)

fig_temp.subplots_adjust(bottom=0.22)

ax_m_dot_slider = fig_temp.add_axes([0.27, 0.13, 0.58, 0.025])
ax_temp_slider = fig_temp.add_axes([0.27, 0.09, 0.58, 0.025])
ax_pressure_slider = fig_temp.add_axes([0.27, 0.05, 0.58, 0.025])

m_dot_slider = Slider(
    ax=ax_m_dot_slider,
    label="Cold m_dot [kg/s]",
    valmin=0.02,
    valmax=1.00,
    valinit=m_dot_cold,
    valstep=0.01
)

cold_temp_slider = Slider(
    ax=ax_temp_slider,
    label="Cold inlet T [degC]",
    valmin=150.0,
    valmax=320.0,
    valinit=T_cold_in_C,
    valstep=1.0
)

cold_pressure_slider = Slider(
    ax=ax_pressure_slider,
    label="Cold inlet P [MPa]",
    valmin=3.0,
    valmax=10.0,
    valinit=P_cold_in / 1e6,
    valstep=0.1
)


def update_interactive_plot(_value):
    try:
        result = calculate_case(
            m_dot_cold_case=m_dot_slider.val,
            P_cold_in_case=cold_pressure_slider.val * 1e6,
            T_cold_in_C_case=cold_temp_slider.val
        )
    except Exception as exc:
        info_text.set_text(f"Could not update case:\n{exc}")
        fig_temp.canvas.draw_idle()
        return

    line_hot.set_ydata(result["T_hot_C"])
    line_cold.set_ydata(result["T_cold_plot_C"])
    line_sat.set_ydata(result["T_sat_cold_C"])
    line_hot_pressure.set_ydata(result["hot_cumulative_dP"])
    line_cold_pressure.set_ydata(result["cold_cumulative_dP_from_inlet"])
    info_text.set_text(result["condition_text"])

    for patch in region_patches:
        patch.remove()
    region_patches.clear()

    boiling_now = result["boiling_indices"]
    if boiling_now.size > 0:
        region_patches.append(
            ax_temp.axvspan(
                x[boiling_now[0]],
                x[boiling_now[-1]],
                color="skyblue",
                alpha=0.18,
                label="Boiling region: Shah correlation"
            )
        )

    superheated_now = result["superheated_indices"]
    if superheated_now.size > 0:
        region_patches.append(
            ax_temp.axvspan(
                x[superheated_now[0]],
                x[superheated_now[-1]],
                color="orange",
                alpha=0.14,
                label="Superheated vapor region"
            )
        )

    ax_temp.relim()
    ax_temp.autoscale_view(scalex=False, scaley=True)
    ax_pressure.relim()
    ax_pressure.autoscale_view(scalex=False, scaley=True)
    fig_temp.canvas.draw_idle()


m_dot_slider.on_changed(update_interactive_plot)
cold_temp_slider.on_changed(update_interactive_plot)
cold_pressure_slider.on_changed(update_interactive_plot)

fig_temp.savefig(
    "steam_generator_summary.png",
    dpi=300,
    bbox_inches="tight"
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
print(f"Hot inlet pressure  at x = 0 : {P_hot[0] / 1e6:.4f} MPa")
print(f"Hot outlet pressure at x = L : {P_hot[-1] / 1e6:.4f} MPa")
print(f"Hot-side pressure drop : {(P_hot[0] - P_hot[-1]) / 1e3:.3f} kPa")

print("\n[Cold side]")
print(f"Cold inlet  at x = L : {T_cold[-1] - 273.15:.2f} °C")
print(f"Cold outlet at x = 0 : {T_cold[0] - 273.15:.2f} °C")
print(f"Cold inlet pressure  at x = L : {P_cold[-1] / 1e6:.4f} MPa")
print(f"Cold outlet pressure at x = 0 : {P_cold[0] / 1e6:.4f} MPa")
print(f"Cold-side pressure drop : {(P_cold[-1] - P_cold[0]) / 1e3:.3f} kPa")
print(f"Cold-side two-phase pressure drop model : {cold_two_phase_dp_model}")
print(f"Cold-side cumulative pressure drop at outlet : {cold_cumulative_dP[0]:.3f} kPa")

print("\n[Phase change]")
print(f"T_sat at cold inlet  : {T_sat_cold[-1] - 273.15:.2f} °C")
print(f"T_sat at cold outlet : {T_sat_cold[0] - 273.15:.2f} °C")
print(f"Latent heat H_fg at cold inlet  : {H_fg_cold[-1] / 1e3:.2f} kJ/kg")
print(f"Latent heat H_fg at cold outlet : {H_fg_cold[0] / 1e3:.2f} kJ/kg")
print(f"Cold outlet quality at x = 0 : {quality[0]:.3f}")
print(f"Cold outlet phase : {phase[0]}")
print(f"Cold outlet superheat : {max(T_cold[0] - T_sat_cold[0], 0.0):.2f} K")

print("\n[Correlation used in cold side]")
print(df_cell["cold_correlation"].value_counts())

if np.max(H_cold - H_g_cold) <= 0.0:
    print(
        "\nNote: Cold side did not reach the superheated vapor region. "
        "Increase L or m_dot_hot, raise T_hot_in, or reduce m_dot_cold "
        "if you want the gas-temperature-rise region to appear."
    )

print("==============================")
print(f"Node CSV saved: {node_csv_path}")
print(f"Cell CSV saved: {cell_csv_path}")
print("Summary figure saved: steam_generator_summary.png")
