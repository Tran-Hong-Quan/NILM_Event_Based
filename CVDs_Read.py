import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from Utilis.NILM_Utilis import (
    plot_to_bw_image_with_gaussian_dots,calc_prms
)
import config

sampling_rate = 1000
frequency = 50
samples_per_cycle = sampling_rate // frequency
interp_factor = 10


#Đọc lại file
data = np.load("CVDs/all_cycles.npz", allow_pickle=True)
U_CVDS = data["U_CVDS"]
I_CVDS = data["I_CVDS"]

import matplotlib.pyplot as plt

batch_size = 20   # số ảnh muốn hiển thị mỗi lần
cols = 5          # số cột trong grid
rows = batch_size // cols

for batch_start in range(0, len(U_CVDS), batch_size):
    batch_end = batch_start + batch_size
    batch_U2 = U_CVDS[batch_start:batch_end]
    batch_Idiff = I_CVDS[batch_start:batch_end]

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()  # chuyển thành list để dễ duyệt

    for idx, (U_CVD, I_CVD) in enumerate(zip(batch_U2, batch_Idiff)):
        ax = axes[idx]

        P_mean = calc_prms(U_CVD, I_CVD)

        # tạo ảnh từ dữ liệu
        img = plot_to_bw_image_with_gaussian_dots(U_CVD, I_CVD, config.IMAGE_SIZE, config.IMAGE_SIZE,config.IMG_DOT_RADIUS,config.IMG_DOT_ALPHA)

        # hiển thị ảnh
        ax.imshow(img, cmap='gray')
        ax.set_title(f"P={P_mean:.2f}W", fontsize=9)
        ax.axis("off")

        # ép subplot thành hình vuông
        ax.set_aspect('equal', adjustable='box')

    # ẩn subplot thừa nếu batch cuối không đủ 20
    for j in range(idx + 1, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    plt.show()
