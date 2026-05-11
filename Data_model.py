# Data_model.py
from CoolProp.CoolProp import PropsSI


# =========================
# 1. Geometry / Wall data
# =========================
D_h = 2.0e-3              # hydraulic diameter [m]
wall_thickness = 1.0e-3   # wall thickness [m]
k_wall = 20.0             # wall thermal conductivity [W/m-K]


# =========================
# 2. Hot side: Helium
# =========================
hot_fluid = "Helium"
T_hot_in = 750.0 + 273.15     # [K]
m_dot_hot = 1.5               # [kg/s]
P_hot = 7.0e6                 # [Pa]


# =========================
# 3. Cold side: CO2
# =========================
cold_fluid = "CarbonDioxide"
T_cold_in = 400.0 + 273.15    # [K]
T_cold_out_target = 550.0 + 273.15  # [K]
m_dot_cold = 5.0              # [kg/s]
P_cold = 20.0e6               # [Pa]


# =========================
# 4. Property function
# =========================
def get_props(fluid, T, P):
    """
    CoolProp를 이용해 유체 물성치를 계산하는 함수

    Parameters
    ----------
    fluid : str
        유체 이름, 예: "Helium", "CarbonDioxide"
    T : float
        온도 [K]
    P : float
        압력 [Pa]

    Returns
    -------
    dict
        rho : density [kg/m3]
        mu  : dynamic viscosity [Pa·s]
        k   : thermal conductivity [W/m-K]
        cp  : specific heat [J/kg-K]
        Pr  : Prandtl number [-]
    """

    rho = PropsSI("D", "T", T, "P", P, fluid)
    mu = PropsSI("V", "T", T, "P", P, fluid)
    k = PropsSI("L", "T", T, "P", P, fluid)
    cp = PropsSI("C", "T", T, "P", P, fluid)
    Pr = PropsSI("PRANDTL", "T", T, "P", P, fluid)

    return {
        "rho": rho,
        "mu": mu,
        "k": k,
        "cp": cp,
        "Pr": Pr
    }


# =========================
# 5. Test
# =========================
if __name__ == "__main__":
    he_props = get_props(hot_fluid, T_hot_in, P_hot)
    co2_props = get_props(cold_fluid, T_cold_in, P_cold)

    print("Helium properties at inlet")
    print(he_props)

    print("\nCO2 properties at inlet")
    print(co2_props)