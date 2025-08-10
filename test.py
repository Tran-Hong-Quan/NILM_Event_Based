import numpy as np
import matplotlib.pyplot as plt
from Utilis.NILM_Utilis import plot_to_bw_image_with_gaussian_dots ,flip_ui_image

# Giả lập dữ liệu U, I sao cho chấm Gaussian nằm ở Q3
U_test = np.linspace(-1, -0.5, 20)  # X thấp -> về trái
I_test = np.linspace(-1, -0.5, 20)  # Y thấp -> về dưới

img_np = plot_to_bw_image_with_gaussian_dots(U_test, I_test, 32, 32, radius=1, alpha=0.8)

# Kiểm tra Q1 trước lật
def quadrant_sums(img):
    h, w = img.shape
    mid_h, mid_w = h // 2, w // 2
    return {
        "Q1": np.sum(img[0:mid_h, mid_w:w]),
        "Q2": np.sum(img[0:mid_h, 0:mid_w]),
        "Q3": np.sum(img[mid_h:h, 0:mid_w]),
        "Q4": np.sum(img[mid_h:h, mid_w:w]),
    }

print("Before flip:", quadrant_sums(img_np))

# Lật để Q1 lớn nhất
img_flipped = flip_ui_image(img_np)

print("After flip:", quadrant_sums(img_flipped))

# Hiển thị ảnh trước và sau lật
fig, axs = plt.subplots(1, 2, figsize=(6, 3))
axs[0].imshow(img_np, cmap='gray')
axs[0].set_title("Before Flip")
axs[1].imshow(img_flipped, cmap='gray')
axs[1].set_title("After Flip")
plt.show()
