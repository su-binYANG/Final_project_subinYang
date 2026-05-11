import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI
from pathlib import Path

from Correlation import dittus_boelter_h, shah_boiling_h


def load_input(filename="input.json"):
    input_path = Path(__file__).with_name(filename)
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)


def calc_area(D):
    return np.pi * D**2 / 4


def calc_perimeter(D):
    return np.pi * D


def overall_U(h_hot, h_cold, D_i, D_o, k_wall):
    R_hot = 1 / h_hot
    R_wall = D_i * np.log(D_o / D_i) / (2 * k_wall)
    R_cold = D_i / (D_o * h_cold)
    return 1 / (R_hot + R_wall + R_cold)


def run_solver(data):
    fluid_hot = data["fluid_hot"]
    fluid_cold = data["fluid_cold"]

    L = data["L"]
    N = data["N"]
    dz = L / N

    D_i = data["D_inner"]
    D_o = data["D_outer"]
    k_wall = data["k_wall"]

    A_i = calc_area(D_i)
    P_i = calc_perimeter(D_i)

    m_dot_hot = data["m_dot_hot"]
    m_dot_cold = data["m_dot_cold"]

    P_hot = data["P_hot"]
    P_cold = data["P_cold"]

    T_hot = np.zeros(N + 1)
    T_cold = np.zeros(N + 1)
    x_cold = np.zeros(N + 1)
    h_cold_arr = np.zeros(N + 1)

    T_hot[0] = data["T_hot_in"]
    T_cold[0] = data["T_cold_in"]

    h_cold_arr[0] = PropsSI("H", "P", P_cold, "T", T_cold[0], fluid_cold)

    G_hot = m_dot_hot / A_i
    G_cold = m_dot_cold / A_i

    for i in range(N):
        T_h = T_hot[i]
        T_c = T_cold[i]
        h_c = h_cold_arr[i]

        T_sat = PropsSI("T", "P", P_cold, "Q", 0, fluid_cold)
        h_f = PropsSI("H", "P", P_cold, "Q", 0, fluid_cold)
        h_g = PropsSI("H", "P", P_cold, "Q", 1, fluid_cold)
        h_fg = h_g - h_f

        rho_h = PropsSI("D", "P", P_hot, "T", T_h, fluid_hot)
        mu_h = PropsSI("V", "P", P_hot, "T", T_h, fluid_hot)
        k_h = PropsSI("L", "P", P_hot, "T", T_h, fluid_hot)
        cp_h = PropsSI("C", "P", P_hot, "T", T_h, fluid_hot)

        h_hot = dittus_boelter_h(
            G=G_hot,
            Dh=D_i,
            k=k_h,
            mu=mu_h,
            cp=cp_h,
            heating=False
        )

        if h_c < h_f:
            rho_c = PropsSI("D", "P", P_cold, "H", h_c, fluid_cold)
            mu_c = PropsSI("V", "P", P_cold, "H", h_c, fluid_cold)
            k_c = PropsSI("L", "P", P_cold, "H", h_c, fluid_cold)
            cp_c = PropsSI("C", "P", P_cold, "H", h_c, fluid_cold)

            h_cold_htc = dittus_boelter_h(
                G=G_cold,
                Dh=D_i,
                k=k_c,
                mu=mu_c,
                cp=cp_c,
                heating=True
            )

            T_c_effective = T_c

        else:
            x = np.clip((h_c - h_f) / h_fg, 0.0, 1.0)
            x_cold[i] = x

            rho_l = PropsSI("D", "P", P_cold, "Q", 0, fluid_cold)
            rho_g = PropsSI("D", "P", P_cold, "Q", 1, fluid_cold)
            mu_l = PropsSI("V", "P", P_cold, "Q", 0, fluid_cold)
            k_l = PropsSI("L", "P", P_cold, "Q", 0, fluid_cold)
            cp_l = PropsSI("C", "P", P_cold, "Q", 0, fluid_cold)

            h_sp = dittus_boelter_h(
                G=G_cold,
                Dh=D_i,
                k=k_l,
                mu=mu_l,
                cp=cp_l,
                heating=True
            )

            q_guess = max(1.0, h_sp * (T_h - T_sat))

            h_cold_htc = shah_boiling_h(
                h_sp=h_sp,
                q_flux=q_guess,
                G=G_cold,
                h_fg=h_fg,
                x=x,
                rho_g=rho_g,
                rho_l=rho_l,
                Dh=D_i
            )

            T_c_effective = T_sat

        U = overall_U(h_hot, h_cold_htc, D_i, D_o, k_wall)

        dA = P_i * dz
        q = U * dA * (T_h - T_c_effective)

        T_hot[i + 1] = T_hot[i] - q / (m_dot_hot * cp_h)

        h_cold_arr[i + 1] = h_cold_arr[i] + q / m_dot_cold

        if h_cold_arr[i + 1] < h_f:
            T_cold[i + 1] = PropsSI("T", "P", P_cold, "H", h_cold_arr[i + 1], fluid_cold)
        else:
            T_cold[i + 1] = T_sat
            x_cold[i + 1] = np.clip((h_cold_arr[i + 1] - h_f) / h_fg, 0.0, 1.0)

    position = np.linspace(0, L, N + 1)

    result = pd.DataFrame({
        "position_m": position,
        "T_hot_K": T_hot,
        "T_hot_C": T_hot - 273.15,
        "T_cold_K": T_cold,
        "T_cold_C": T_cold - 273.15,
        "h_cold_J_kg": h_cold_arr,
        "x_cold": x_cold
    })

    return result


def plot_temperature(result):
    plt.figure(figsize=(8, 5))

    plt.plot(
        result["position_m"],
        result["T_hot_C"],
        label="Hot side"
    )

    plt.plot(
        result["position_m"],
        result["T_cold_C"],
        label="Cold side"
    )

    plt.xlabel("Position [m]")
    plt.ylabel("Temperature [°C]")
    plt.title("Temperature Distribution along Heat Exchanger")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig("position-temperature.png", dpi=300)
    plt.show()


def main():
    data = load_input("input.json")
    result = run_solver(data)

    result.to_csv("node_result.csv", index=False, encoding="utf-8-sig")

    print(result)
    print("\nCSV saved: node_result.csv")
    print("Figure saved: position-temperature.png")

    plot_temperature(result)


if __name__ == "__main__":
    main()
