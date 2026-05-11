# Nodal_solver.py

import pandas as pd

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

from Physics_engine import (
    calc_fluid_h,
    calc_overall_U
)


def run_simulation(L, N=100, save_csv=False):
    """
    대향류 열교환기 nodal 해석

    Parameters
    ----------
    L : float
        열교환기 길이 [m]

    N : int
        node 개수

    save_csv : bool
        True이면 csv 저장

    Returns
    -------
    T_cold_out : float
        CO2 출구온도 [K]
    df : pandas.DataFrame
        node별 계산 결과
    """

    dx = L / N

    # 원형 유로 기준 열전달 면적
    perimeter = 3.141592 * D_h
    dA = perimeter * dx

    # 온도 초기값
    T_hot = T_hot_in
    T_cold = T_cold_in

    results = []

    for i in range(N):
        # 현재 node 평균 온도 기준 물성 계산
        hot_props = get_props(hot_fluid, T_hot, P_hot)
        cold_props = get_props(cold_fluid, T_cold, P_cold)

        # h 계산
        hot_h_data = calc_fluid_h(m_dot_hot, hot_props, D_h)
        cold_h_data = calc_fluid_h(m_dot_cold, cold_props, D_h)

        h_hot = hot_h_data["h"]
        h_cold = cold_h_data["h"]

        # 전체 열전달계수
        U = calc_overall_U(
            h_hot,
            h_cold,
            wall_thickness,
            k_wall
        )

        # 온도차
        dT = T_hot - T_cold

        # node 열전달량
        dQ = U * dA * dT

        # 온도 변화
        dT_hot = dQ / (m_dot_hot * hot_props["cp"])
        dT_cold = dQ / (m_dot_cold * cold_props["cp"])

        # hot side는 냉각
        T_hot_new = T_hot - dT_hot

        # cold side는 가열
        T_cold_new = T_cold + dT_cold

        results.append({
            "node": i + 1,
            "x_m": (i + 1) * dx,
            "T_hot_C": T_hot_new - 273.15,
            "T_cold_C": T_cold_new - 273.15,
            "dQ_W": dQ,
            "U_W_m2K": U,
            "h_hot_W_m2K": h_hot,
            "h_cold_W_m2K": h_cold,
            "Re_hot": hot_h_data["Re"],
            "Re_cold": cold_h_data["Re"]
        })

        T_hot = T_hot_new
        T_cold = T_cold_new

    df = pd.DataFrame(results)

    if save_csv:
        df.to_csv("nodal_result.csv", index=False, encoding="utf-8-sig")

    return T_cold, df


if __name__ == "__main__":
    L_test = 1.0

    T_cold_out, df = run_simulation(
        L=L_test,
        N=100,
        save_csv=True
    )

    print("===== Nodal Solver Result =====")
    print(f"Length = {L_test:.3f} m")
    print(f"CO2 outlet temperature = {T_cold_out - 273.15:.2f} °C")
    print(df.tail())