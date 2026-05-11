# Optimizer.py

from Data_model import T_cold_out_target
from Nodal_Solver import run_simulation


def optimize_length(
    L_min=0.1,
    L_max=50.0,
    tol_T=0.1,
    max_iter=100,
    N=100
):
    """
    CO2 출구온도가 목표 온도에 도달하도록
    열교환기 길이 L을 찾는 함수

    Parameters
    ----------
    L_min : float
        최소 길이 [m]

    L_max : float
        최대 길이 [m]

    tol_T : float
        온도 허용 오차 [K]

    max_iter : int
        최대 반복 횟수

    N : int
        node 개수

    Returns
    -------
    L_mid : float
        최적 열교환기 길이 [m]

    T_cold_out : float
        최종 CO2 출구온도 [K]

    df : DataFrame
        node별 결과
    """

    for i in range(max_iter):
        L_mid = 0.5 * (L_min + L_max)

        T_cold_out, df = run_simulation(
            L=L_mid,
            N=N,
            save_csv=False
        )

        error = T_cold_out - T_cold_out_target

        print(
            f"Iter {i+1:03d} | "
            f"L = {L_mid:.4f} m | "
            f"T_CO2_out = {T_cold_out - 273.15:.2f} °C | "
            f"Error = {error:.4f} K"
        )

        if abs(error) < tol_T:
            print("\nConverged!")
            return L_mid, T_cold_out, df

        if T_cold_out < T_cold_out_target:
            # CO2 온도가 목표보다 낮음 → 열교환 부족 → 길이 증가
            L_min = L_mid
        else:
            # CO2 온도가 목표보다 높음 → 열교환 과다 → 길이 감소
            L_max = L_mid

    print("\nMax iteration reached.")
    return L_mid, T_cold_out, df


if __name__ == "__main__":

    L_opt, T_out, df_result = optimize_length(
        L_min=0.1,
        L_max=50.0,
        tol_T=0.1,
        max_iter=100,
        N=100
    )

    df_result.to_csv(
        "optimized_nodal_result.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\n===== Optimization Result =====")
    print(f"Optimal length = {L_opt:.4f} m")
    print(f"CO2 outlet temperature = {T_out - 273.15:.2f} °C")
    print("CSV saved: optimized_nodal_result.csv")