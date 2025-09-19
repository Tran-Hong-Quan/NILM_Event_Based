import pandas as pd
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from Utilis.NILM_Utilis import (
    CycleInterpolator, align_phase, calc_prms, 
    plot_to_bw_image_with_gaussian_dots, flip_ui_image, is_right_side_greater
)

# --- Cấu hình ---
maxCvdCounts = 50
csv_path = r"ElectricDatas\MyNewData\NO\sacmt_event_no.csv"
parts = csv_path.replace("\\", "/").split("/")
csv_path = os.path.join(*parts)
sampling_rate = 1000
frequency = 50
samples_per_cycle = sampling_rate // frequency
test_cycles = 100
interp_factor = 10
SAMPLE_PER_IMAGE = test_cycles * samples_per_cycle

# --- Đọc dữ liệu ---
df = pd.read_csv(csv_path)
Power = df["Power"].values
I_raw = df["In"].values
U_raw = df["Un"].values
print("Số mẫu dữ liệu:", len(I_raw))

total_cycles = len(I_raw) // samples_per_cycle


def process_pair(delay1, delay2, I_raw, U_raw):
    start1 = delay1 * samples_per_cycle
    end1 = start1 + SAMPLE_PER_IMAGE
    interp1 = CycleInterpolator(samples_per_cycle, interp_factor)
    interp1.update_batch(I_raw[start1:end1], U_raw[start1:end1])
    U1, I1 = interp1.get_average()

    start2 = delay2 * samples_per_cycle
    end2 = start2 + SAMPLE_PER_IMAGE
    interp2 = CycleInterpolator(samples_per_cycle, interp_factor)
    interp2.update_batch(I_raw[start2:end2], U_raw[start2:end2])
    U2, I2 = interp2.get_average()

    U1_aligned, best_shift = align_phase(U2, U1)
    I1_aligned = np.roll(I1, -best_shift)
    I_diff = (I2 - I1_aligned)
    I_diff *= is_right_side_greater(I_diff, U2)

    delta_P_mean = calc_prms(U2, I_diff)
    if delta_P_mean > 7:
        img = plot_to_bw_image_with_gaussian_dots(U2, I_diff, 32, 32, 2, 0.3)
        img = flip_ui_image(img)
        return U2, I_diff
    return None


if __name__ == "__main__":
    # --- Gom các cặp ---
    pairs = [
        (d1, d2)
        for d1 in range(0, total_cycles - test_cycles, test_cycles)
        for d2 in range(d1 + test_cycles, total_cycles - test_cycles, test_cycles)
    ]

    U_CVDS, I_CVDS = [], []

    # --- Chạy song song ---
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_pair, d1, d2, I_raw, U_raw) for d1, d2 in pairs]

        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                U_CVDS.append(result[0])
                I_CVDS.append(result[1])
                print("Số cvds =", len(U_CVDS))
                if len(U_CVDS) >= maxCvdCounts:
                    break

    # --- Lưu ---
    os.makedirs("CVDs", exist_ok=True)
    np.savez("CVDs/all_cycles2.npz", U_CVDS=U_CVDS, I_CVDS=I_CVDS)
