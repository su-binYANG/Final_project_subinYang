# correlation.py
"""
Boiling heat transfer and two-phase pressure drop correlations

Included correlations:
    - Chen
    - Shah
    - Gungor & Winterton
    - Bertsch et al.
    - Kim & Mudawar
    - Jens & Lottes
    - Homogeneous Equilibrium Model pressure drop
    - Lockhart-Martinelli separated flow pressure drop
    - Friedel pressure drop
    - Muller-Steinhagen & Heck pressure drop

All units are SI.
"""

import math


G_CONST = 9.81
EPS = 1.0e-12


# ============================================================
# Utility
# ============================================================

def clip_quality(x, eps=1.0e-6):
    """Avoid x = 0 or x = 1 singularity."""
    return max(eps, min(1.0 - eps, x))


def safe_positive(value, name="value"):
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


# ============================================================
# Single-phase heat transfer
# ============================================================

def dittus_boelter_h(G, Dh, k, mu, cp, heating=True):
    """
    Dittus-Boelter correlation

    Nu = 0.023 Re^0.8 Pr^n
    n = 0.4 for heating
    n = 0.3 for cooling
    """
    Re = G * Dh / mu
    Pr = cp * mu / k

    n = 0.4 if heating else 0.3
    Nu = 0.023 * Re**0.8 * Pr**n
    h = Nu * k / Dh

    return {
        "h": h,
        "Nu": Nu,
        "Re": Re,
        "Pr": Pr,
    }


def hausen_h(G, Dh, L, k, mu, cp):
    """
    Hausen correlation

    Used in Bertsch correlation for h_sp,lo and h_sp,go.
    """
    Re = G * Dh / mu
    Pr = cp * mu / k

    Nu = 3.66 + (
        0.0668 * (Dh / L) * Re * Pr
    ) / (
        1.0 + 0.04 * ((Dh / L) * Re * Pr)**(2.0 / 3.0)
    )

    h = Nu * k / Dh

    return {
        "h": h,
        "Nu": Nu,
        "Re": Re,
        "Pr": Pr,
    }


# ============================================================
# Dimensionless numbers
# ============================================================

def boiling_number(q_flux, G, h_fg):
    return q_flux / (G * h_fg)


def froude_liquid(G, rho_l, Dh):
    return G**2 / (rho_l**2 * G_CONST * Dh)


def weber_number(G, Dh, rho, sigma):
    return G**2 * Dh / (rho * sigma)


def martinelli_xtt(x, rho_g, rho_l, mu_l, mu_g):
    """
    X_tt = ((1-x)/x)^0.9 * (rho_g/rho_l)^0.5 * (mu_l/mu_g)^0.1
    """
    x = clip_quality(x)

    return (
        ((1.0 - x) / x)**0.9
        * (rho_g / rho_l)**0.5
        * (mu_l / mu_g)**0.1
    )


def shah_convection_number(x, rho_g, rho_l, G, Dh):
    """
    Shah convection number with low-Froude correction.
    """
    x = clip_quality(x)

    Fr_l = froude_liquid(G, rho_l, Dh)

    Co = (
        ((1.0 - x) / x)**0.8
        * (rho_g / rho_l)**0.5
    )

    if Fr_l < 0.04:
        N = 0.38 * Fr_l**(-0.3) * Co
    else:
        N = Co

    return {
        "Co": Co,
        "N": N,
        "Fr_l": Fr_l,
    }


# ============================================================
# Pool boiling
# ============================================================

def cooper_pool_boiling_h(P_r, molecular_weight, q_flux):
    """
    Cooper pool boiling correlation

    h_pool = 55 * P_r^0.12 * (-log10(P_r))^-0.55
             * M^-0.5 * q''^0.67
    """
    if P_r <= 0 or P_r >= 1:
        raise ValueError("P_r must be between 0 and 1.")

    h_pool = (
        55.0
        * P_r**0.12
        * (-math.log10(P_r))**(-0.55)
        * molecular_weight**(-0.5)
        * q_flux**0.67
    )

    return h_pool


def forster_zuber_h(
    k_l,
    cp_l,
    rho_l,
    rho_v,
    mu_l,
    sigma,
    h_fg,
    delta_T,
    delta_P,
    g_c=1.0,
    S=1.0,
):
    """
    Forster-Zuber nucleate boiling term

    h_mic = 0.00122 *
            k_l^0.79 cp_l^0.45 rho_l^0.49 g_c^0.25
            / (sigma^0.5 mu_l^0.29 h_fg^0.24 rho_v^0.24)
            * delta_T^0.24 * delta_P^0.75 * S
    """
    h = (
        0.00122
        * k_l**0.79
        * cp_l**0.45
        * rho_l**0.49
        * g_c**0.25
        * delta_T**0.24
        * delta_P**0.75
        * S
        / (
            sigma**0.5
            * mu_l**0.29
            * h_fg**0.24
            * rho_v**0.24
        )
    )

    return h


# ============================================================
# Chen correlation
# ============================================================

def chen_correlation(h_db, h_fz, Re_tp, Re_l, S=1.0):
    """
    Chen correlation

    h_tp = h_mac + h_mic
    h_mac = h_DB * F
    F = (Re_tp / Re_l)^0.8
    h_mic = h_FZ * S
    """
    Re_l = safe_positive(Re_l, "Re_l")

    F = (Re_tp / Re_l)**0.8

    h_mac = h_db * F
    h_mic = h_fz * S
    h_tp = h_mac + h_mic

    return {
        "h_tp": h_tp,
        "h_mac": h_mac,
        "h_mic": h_mic,
        "F": F,
        "S": S,
    }


# ============================================================
# Shah correlation
# ============================================================

def shah_correlation(h_sp, q_flux, G, h_fg, x, rho_g, rho_l, Dh):
    """
    Shah correlation

    h_tp = psi * h_sp
    psi = max(psi_nb, psi_cb, psi_bs)
    """
    x = clip_quality(x)

    Bo = boiling_number(q_flux, G, h_fg)
    shah_N = shah_convection_number(x, rho_g, rho_l, G, Dh)

    N = max(shah_N["N"], EPS)

    if Bo > 0.3e-4:
        psi_nb = 230.0 * Bo**0.5
    else:
        psi_nb = 1.0 + 46.0 * Bo**0.5

    psi_cb = 1.8 / N**0.8

    if N > 1.0:
        psi_bs = psi_nb
    elif 0.1 < N <= 1.0:
        F = 14.7 if Bo >= 11e-4 else 15.43
        psi_bs = F * Bo**0.5 * math.exp(2.74 * N**(-0.1))
    else:
        F = 14.7 if Bo >= 11e-4 else 15.43
        psi_bs = F * Bo**0.5 * math.exp(2.74 * N**(-0.15))

    psi = max(psi_nb, psi_cb, psi_bs)
    h_tp = psi * h_sp

    return {
        "h_tp": h_tp,
        "psi": psi,
        "psi_nb": psi_nb,
        "psi_cb": psi_cb,
        "psi_bs": psi_bs,
        "Bo": Bo,
        "Co": shah_N["Co"],
        "N": shah_N["N"],
        "Fr_l": shah_N["Fr_l"],
    }


# ============================================================
# Gungor & Winterton correlation
# ============================================================

def gungor_winterton_correlation(
    h_sp,
    h_pool,
    q_flux,
    G,
    h_fg,
    x,
    rho_g,
    rho_l,
    mu_l,
    mu_g,
    Dh,
    horizontal=False,
):
    """
    Gungor & Winterton correlation

    h_tp = E*h_sp + S*h_pool
    """
    x = clip_quality(x)

    Bo = boiling_number(q_flux, G, h_fg)
    Xtt = martinelli_xtt(x, rho_g, rho_l, mu_l, mu_g)

    Re_l = G * (1.0 - x) * Dh / mu_l

    E = 1.0 + 24000.0 * Bo**1.16 + 1.37 * (1.0 / Xtt)**0.86
    S = 1.0 / (1.0 + 1.15e-6 * E**2 * Re_l**1.17)

    Fr_l = froude_liquid(G, rho_l, Dh)

    if horizontal and Fr_l < 0.05:
        E *= Fr_l**(0.1 - 2.0 * Fr_l)
        S *= math.sqrt(Fr_l)

    h_tp = E * h_sp + S * h_pool

    return {
        "h_tp": h_tp,
        "E": E,
        "S": S,
        "Bo": Bo,
        "Xtt": Xtt,
        "Re_l": Re_l,
        "Fr_l": Fr_l,
    }


# ============================================================
# Bertsch et al. correlation
# ============================================================

def bertsch_correlation(
    h_sp_lo,
    h_sp_go,
    h_nb,
    x,
    sigma,
    rho_l,
    rho_g,
    Dh,
):
    """
    Bertsch et al. correlation

    h_tp = E*h_cb + S*h_nb
    h_cb = h_sp,lo*(1-x) + h_sp,go*x
    E = 1 + 80*(x^2 - x^6)*exp(-0.6*Co)
    Co = sqrt(sigma / (g*(rho_l-rho_g)*Dh^2))
    S = 1 - x
    """
    x = clip_quality(x)

    h_cb = h_sp_lo * (1.0 - x) + h_sp_go * x

    Co = math.sqrt(
        sigma / (G_CONST * (rho_l - rho_g) * Dh**2)
    )

    E = 1.0 + 80.0 * (x**2 - x**6) * math.exp(-0.6 * Co)
    S = 1.0 - x

    h_tp = E * h_cb + S * h_nb

    return {
        "h_tp": h_tp,
        "h_cb": h_cb,
        "h_nb": h_nb,
        "E": E,
        "S": S,
        "Co": Co,
    }


# ============================================================
# Kim & Mudawar correlation
# ============================================================

def kim_mudawar_correlation(
    h_db,
    q_flux,
    G,
    h_fg,
    P,
    P_crit,
    x,
    rho_g,
    rho_l,
    mu_l,
    mu_g,
    sigma,
    Dh,
    P_H,
    P_F,
):
    """
    Kim & Mudawar correlation

    h_tp = sqrt(h_nb^2 + h_cb^2)
    """
    x = clip_quality(x)

    Bo = boiling_number(q_flux, G, h_fg)
    PR = P / P_crit
    We_fo = weber_number(G, Dh, rho_l, sigma)
    Xtt = martinelli_xtt(x, rho_g, rho_l, mu_l, mu_g)

    perimeter_ratio = P_H / P_F

    h_cb = (
        2345.0
        * (Bo * perimeter_ratio)**0.7
        * PR**0.38
        * (1.0 - x)**(-0.51)
        * h_db
    )

    h_nb = (
        5.2
        * (Bo * perimeter_ratio)**0.08
        * We_fo**(-0.54)
        + 3.5
        * (1.0 / Xtt)**0.94
        * (rho_g / rho_l)**0.25
    ) * h_db

    h_tp = math.sqrt(h_nb**2 + h_cb**2)

    return {
        "h_tp": h_tp,
        "h_nb": h_nb,
        "h_cb": h_cb,
        "Bo": Bo,
        "PR": PR,
        "We_fo": We_fo,
        "Xtt": Xtt,
        "P_H_over_P_F": perimeter_ratio,
    }


# ============================================================
# Jens & Lottes correlation
# ============================================================

def jens_lottes_delta_T(q_flux, P):
    """
    Jens & Lottes correlation, SI unit

    delta_T [K] = 0.7925 * q''^0.25 * exp(P / 28728)
    q'' [W/m2]
    P   [Pa]
    """
    return 0.7925 * q_flux**0.25 * math.exp(P / 28728.0)


def jens_lottes_h(q_flux, P):
    """
    h = q'' / delta_T
    """
    delta_T = jens_lottes_delta_T(q_flux, P)
    h = q_flux / delta_T

    return {
        "h": h,
        "delta_T": delta_T,
    }


# ============================================================
# Friction factor
# ============================================================

def churchill_friction_factor(Re, roughness=0.0, Dh=1.0):
    """
    Churchill Darcy friction factor
    """
    Re = safe_positive(Re, "Re")

    eps_D = roughness / Dh

    A = (
        2.457
        * math.log(
            1.0 / ((7.0 / Re)**0.9 + 0.27 * eps_D)
        )
    )**16

    B = (37530.0 / Re)**16

    f = 8.0 * (
        (8.0 / Re)**12
        + 1.0 / (A + B)**1.5
    )**(1.0 / 12.0)

    return f


# ============================================================
# Homogeneous Equilibrium Model pressure drop
# ============================================================

def hem_mixture_density(x, rho_l, rho_v):
    """
    rho_m = [x/rho_v + (1-x)/rho_l]^-1
    """
    x = clip_quality(x)
    return 1.0 / (x / rho_v + (1.0 - x) / rho_l)


def hem_mixture_viscosity(x, mu_l, mu_v, model="McAdams"):
    """
    Mixture viscosity models:
        McAdams:
            mu_tp = [x/mu_v + (1-x)/mu_l]^-1

        Cicchitti:
            mu_tp = x*mu_v + (1-x)*mu_l
    """
    x = clip_quality(x)
    model = model.lower()

    if model == "mcadams":
        return 1.0 / (x / mu_v + (1.0 - x) / mu_l)

    if model == "cicchitti":
        return x * mu_v + (1.0 - x) * mu_l

    raise ValueError("model must be 'McAdams' or 'Cicchitti'.")


def dukler_viscosity(x, rho_m, rho_l, rho_v, mu_l, mu_v):
    """
    Dukler mixture viscosity
    """
    x = clip_quality(x)

    return rho_m * (
        x * mu_v / rho_v
        + (1.0 - x) * mu_l / rho_l
    )


def hem_pressure_drop(
    G,
    Dh,
    x,
    rho_l,
    rho_v,
    mu_l,
    mu_v,
    viscosity_model="McAdams",
    roughness=0.0,
):
    """
    Homogeneous Equilibrium Model

    dP/dz = f_tp * G^2 / (2*rho_m*Dh)
    """
    rho_m = hem_mixture_density(x, rho_l, rho_v)

    if viscosity_model.lower() == "dukler":
        mu_tp = dukler_viscosity(x, rho_m, rho_l, rho_v, mu_l, mu_v)
    else:
        mu_tp = hem_mixture_viscosity(x, mu_l, mu_v, viscosity_model)

    Re_tp = G * Dh / mu_tp
    f_tp = churchill_friction_factor(Re_tp, roughness, Dh)

    dpdz = f_tp * G**2 / (2.0 * rho_m * Dh)

    return {
        "dpdz": dpdz,
        "rho_m": rho_m,
        "mu_tp": mu_tp,
        "Re_tp": Re_tp,
        "f_tp": f_tp,
    }


# ============================================================
# Separated flow pressure drop
# ============================================================

def lockhart_martinelli_phi2(X, C=20.0):
    """
    phi_f^2 = 1 + C/X + 1/X^2

    Cvv = 5
    Ctv = 10
    Cvt = 12
    Ctt = 20
    """
    X = safe_positive(X, "X")
    return 1.0 + C / X + 1.0 / X**2


def lockhart_martinelli_pressure_drop(dpdz_f, X, C=20.0):
    """
    dP/dz_TP = dP/dz_f * phi_f^2
    """
    phi2 = lockhart_martinelli_phi2(X, C)
    dpdz_tp = dpdz_f * phi2

    return {
        "dpdz_tp": dpdz_tp,
        "phi2": phi2,
    }


def friedel_pressure_drop(
    dpdz_fo,
    x,
    rho_l,
    rho_g,
    mu_l,
    mu_g,
    f_fo,
    f_go,
    G,
    Dh,
    sigma,
):
    """
    Friedel pressure drop correlation

    dP/dz_TP = dP/dz_fo * phi_fo^2
    """
    x = clip_quality(x)

    rho_m = hem_mixture_density(x, rho_l, rho_g)

    Fr_tp = G**2 / (G_CONST * Dh * rho_m**2)
    We_tp = G**2 * Dh / (sigma * rho_m)

    phi2 = (
        (1.0 - x)**2
        + x**2 * (rho_l / rho_g) * (f_go / f_fo)
        + 3.24
        * x**0.78
        * (1.0 - x)**0.224
        * (rho_l / rho_g)**0.91
        * (mu_g / mu_l)**0.19
        * (1.0 - mu_g / mu_l)**0.7
        * Fr_tp**(-0.045)
        * We_tp**(-0.035)
    )

    dpdz_tp = dpdz_fo * phi2

    return {
        "dpdz_tp": dpdz_tp,
        "phi2": phi2,
        "rho_m": rho_m,
        "Fr_tp": Fr_tp,
        "We_tp": We_tp,
    }


def muller_steinhagen_heck_pressure_drop(dpdz_fo, dpdz_go, x):
    """
    Muller-Steinhagen & Heck correlation

    dP/dz_TP =
        [dP/dz_fo + 2(dP/dz_go - dP/dz_fo)x] * (1-x)^(1/3)
        + dP/dz_go*x^3
    """
    x = clip_quality(x)

    dpdz_tp = (
        (dpdz_fo + 2.0 * (dpdz_go - dpdz_fo) * x)
        * (1.0 - x)**(1.0 / 3.0)
        + dpdz_go * x**3
    )

    return {
        "dpdz_tp": dpdz_tp,
    }


# ============================================================
# Micro-channel separated flow pressure drop
# ============================================================

def mishima_hibiki_C(Dh, geometry="circular"):
    """
    Mishima & Hibiki C value

    Circular:
        C = 21[1 - exp(-0.333 Dh)]

    Rectangular:
        C = 21[1 - exp(-0.319 Dh)]

    Important:
        In the original empirical form, Dh is often inserted in mm.
        Check unit consistency before use.
    """
    geometry = geometry.lower()

    if geometry == "circular":
        return 21.0 * (1.0 - math.exp(-0.333 * Dh))

    if geometry == "rectangular":
        return 21.0 * (1.0 - math.exp(-0.319 * Dh))

    raise ValueError("geometry must be 'circular' or 'rectangular'.")


def yu_microchannel_phi2(x, rho_g, rho_l, Re_g, Re_l):
    """
    Yu et al. micro-channel two-phase multiplier

    phi_f^2 =
        18.65 * (rho_g/rho_l)^0.5
        * ((1-x)/x)
        * Re_g^0.1
        * Re_l^0.5
        - 1.9
    """
    x = clip_quality(x)

    phi2 = (
        18.65
        * (rho_g / rho_l)**0.5
        * ((1.0 - x) / x)
        * Re_g**0.1
        * Re_l**0.5
        - 1.9
    )

    return phi2


def sun_mishima_phi2(X, Re_l, Re_g, x, N_conf):
    """
    Sun & Mishima two-phase multiplier
    """
    x = clip_quality(x)
    X = safe_positive(X, "X")

    if Re_l < 2000.0 and Re_g < 2000.0:
        C = (
            26.0
            * (1.0 + Re_l / 1000.0)
            * (
                1.0
                - math.exp(
                    -0.153 / (0.27 * N_conf + 0.8)
                )
            )
        )

        phi2 = 1.0 + C / X + 1.0 / X**2

    else:
        C = (
            1.79
            * (Re_g / Re_l)**0.4
            * ((1.0 - x) / x)**0.5
        )

        phi2 = 1.0 + C / X**1.19 + 1.0 / X**2

    return {
        "phi2": phi2,
        "C": C,
    }