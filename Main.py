# Main.py

import matplotlib.pyplot as plt

from Data_model import T_cold_out_target
from Optimizer import optimize_length


def main():

    print("===== Heat Exchanger Optimization Start =====")

    L_opt, T_out, df_result = optimize_length(
        L_min=0.1,
        L_max=50.0,
        tol_T=0.1,
        max_iter=100,
        N=100
    )

    df_result.to_csv(
        "final_result.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\n===== Final Result =====")
    print(f"Optimal Heat Exchanger Length = {L_opt:.4f} m")
    print(f"Target CO2 Outlet Temperature = {T_cold_out_target - 273.15:.2f} °C")
    print(f"Calculated CO2 Outlet Temperature = {T_out - 273.15:.2f} °C")
    print("CSV saved: final_result.csv")

    # =========================
    # Temperature Profile Plot
    # =========================
    plt.figure(figsize=(8, 5))
    plt.plot(
        df_result["x_m"],
        df_result["T_hot_C"],
        label="Helium Temperature"
    )
    plt.plot(
        df_result["x_m"],
        df_result["T_cold_C"],
        label="CO2 Temperature"
    )

    plt.xlabel("Heat Exchanger Length [m]")
    plt.ylabel("Temperature [°C]")
    plt.title("Temperature Profile in Counter-Current Heat Exchanger")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("temperature_profile.png", dpi=300)
    plt.show()

    # =========================
    # Heat Transfer Rate Plot
    # =========================
    plt.figure(figsize=(8, 5))
    plt.plot(
        df_result["x_m"],
        df_result["dQ_W"],
        label="Heat Transfer Rate per Node"
    )

    plt.xlabel("Heat Exchanger Length [m]")
    plt.ylabel("dQ [W]")
    plt.title("Heat Transfer Rate Distribution")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("heat_transfer_rate.png", dpi=300)
    plt.show()

    # =========================
    # Overall U Plot
    # =========================
    plt.figure(figsize=(8, 5))
    plt.plot(
        df_result["x_m"],
        df_result["U_W_m2K"],
        label="Overall Heat Transfer Coefficient"
    )

    plt.xlabel("Heat Exchanger Length [m]")
    plt.ylabel("U [W/m²·K]")
    plt.title("Overall Heat Transfer Coefficient Distribution")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("overall_U.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()