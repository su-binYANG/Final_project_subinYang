# Physics_engine.py

import math


def calc_area(D_h):
    """
    유로 단면적 계산
    원형 관 기준
    """
    return math.pi * D_h**2 / 4


def calc_velocity(m_dot, rho, D_h):
    """
    유속 계산
    m_dot : 질량유량 [kg/s]
    rho   : 밀도 [kg/m3]
    D_h   : 수력직경 [m]
    """
    A = calc_area(D_h)
    velocity = m_dot / (rho * A)
    return velocity


def calc_reynolds(rho, velocity, D_h, mu):
    """
    Reynolds number 계산
    """
    Re = rho * velocity * D_h / mu
    return Re


def calc_nusselt(Re, Pr):
    """
    Nusselt number 계산

    난류 조건: Dittus-Boelter 식 사용
    Nu = 0.023 Re^0.8 Pr^0.4

    층류 조건: 원형관 완전발달 유동 가정
    Nu = 3.66
    """
    if Re < 2300:
        Nu = 3.66
    else:
        Nu = 0.023 * (Re ** 0.8) * (Pr ** 0.4)

    return Nu


def calc_heat_transfer_coefficient(Nu, k, D_h):
    """
    대류 열전달계수 h 계산
    """
    h = Nu * k / D_h
    return h


def calc_overall_U(h_hot, h_cold, wall_thickness, k_wall):
    """
    전체 열전달계수 U 계산

    1/U = 1/h_hot + wall_thickness/k_wall + 1/h_cold
    """
    R_total = (1 / h_hot) + (wall_thickness / k_wall) + (1 / h_cold)
    U = 1 / R_total
    return U


def calc_fluid_h(m_dot, props, D_h):
    """
    한 유체의 Re, Nu, h 계산

    props는 Data_model.py의 get_props() 결과 사용
    props = {
        "rho": rho,
        "mu": mu,
        "k": k,
        "cp": cp,
        "Pr": Pr
    }
    """
    rho = props["rho"]
    mu = props["mu"]
    k = props["k"]
    Pr = props["Pr"]

    velocity = calc_velocity(m_dot, rho, D_h)
    Re = calc_reynolds(rho, velocity, D_h, mu)
    Nu = calc_nusselt(Re, Pr)
    h = calc_heat_transfer_coefficient(Nu, k, D_h)

    return {
        "velocity": velocity,
        "Re": Re,
        "Nu": Nu,
        "h": h
    }


if __name__ == "__main__":
    from Data_model import (
        D_h,
        wall_thickness,
        k_wall,
        hot_fluid,
        cold_fluid,
        T_hot_in,
        T_cold_in,
        P_hot,
        P_cold,
        m_dot_hot,
        m_dot_cold,
        get_props
    )

    hot_props = get_props(hot_fluid, T_hot_in, P_hot)
    cold_props = get_props(cold_fluid, T_cold_in, P_cold)

    hot_result = calc_fluid_h(m_dot_hot, hot_props, D_h)
    cold_result = calc_fluid_h(m_dot_cold, cold_props, D_h)

    U = calc_overall_U(
        hot_result["h"],
        cold_result["h"],
        wall_thickness,
        k_wall
    )

    print("===== Hot side: Helium =====")
    print(hot_result)

    print("\n===== Cold side: CO2 =====")
    print(cold_result)

    print("\n===== Overall heat transfer coefficient =====")
    print(f"U = {U:.2f} W/m2-K")