import numpy as np
import matplotlib.pyplot as plt
from Utilis.NILM_Utilis import plot_to_bw_image_with_gaussian_dots ,flip_ui_image
from PIL import Image
import numpy as np
def load_image_from_path(path: str, grayscale=True) -> np.ndarray:
    # Mở ảnh bằng PIL
    img = Image.open(path)

    # Chuyển sang grayscale nếu muốn
    if grayscale:
        img = img.convert("L")

    # Chuyển sang numpy array
    arr = np.array(img)

    return arr

import numpy as np

def flip_to_max_top_right(img: np.ndarray):
    """
    img: ảnh xám 2D dạng numpy (uint8/uint16 hoặc float [0..1] hay [0..255])
    Trả về:
      - img_out: ảnh đã lật sao cho 'độ đen' ở góc trên-phải là lớn nhất
      - darkness: dict độ đen tuyệt đối của từng góc {'TL','TR','BL','BR'}
      - darkness_pct: dict % độ đen của từng góc
      - max_corner: góc có độ đen lớn nhất trước khi lật
    """
    if img.ndim != 2:
        raise ValueError("Ảnh phải là mảng 2D (grayscale).")

    h, w = img.shape
    h2, w2 = h // 2, w // 2

    # Xác định giá trị max có thể (để tính 'mức đen' = max_val - pixel)
    if np.issubdtype(img.dtype, np.integer):
        max_val = np.iinfo(img.dtype).max
    else:
        max_val = 1.0 if img.max() <= 1.0 else 255.0

    # Cắt 4 góc
    TL = img[:h2, :w2]
    TR = img[:h2, w - w2:]
    BL = img[h - h2:, :w2]
    BR = img[h - h2:, w - w2:]

    # Hàm tính độ đen
    def darkness_sum(patch):
        return float(np.sum(max_val - patch.astype(np.float64)))

    darkness = {
        'TL': darkness_sum(TL),
        'TR': darkness_sum(TR),
        'BL': darkness_sum(BL),
        'BR': darkness_sum(BR),
    }

    total_darkness = sum(darkness.values())
    darkness_pct = {k: (v / total_darkness * 100.0) if total_darkness > 0 else 0.0
                    for k, v in darkness.items()}

    # Góc đen nhất hiện tại
    max_corner = max(darkness, key=darkness.get)

    # In ra % độ đen
    print("Độ đen theo %:")
    for k in ['TL', 'TR', 'BL', 'BR']:
        print(f"  {k}: {darkness_pct[k]:.2f}%")

    # Lật ảnh để đưa góc đen nhất lên trên-phải
    if max_corner == 'TR':
        img_out = img.copy()
    elif max_corner == 'TL':
        img_out = np.fliplr(img)
    elif max_corner == 'BR':
        img_out = np.flipud(img)
    else:  # 'BL'
        img_out = np.fliplr(np.flipud(img))

    return img_out, darkness, max_corner


def flip_image(img: np.ndarray, axis: str) -> np.ndarray:
    """
    Lật ảnh numpy theo trục.
    
    axis:
      - 'horizontal' : lật trái ↔ phải
      - 'vertical'   : lật trên ↔ dưới
      - 'both'       : lật cả ngang lẫn dọc
    """
    if axis == 'horizontal':
        return np.flip(img, axis=1)  # lật theo cột
    elif axis == 'vertical':
        return np.flip(img, axis=0)  # lật theo hàng
    elif axis == 'both':
        return np.flipud(np.fliplr(img))
    else:
        raise ValueError("axis phải là 'horizontal', 'vertical' hoặc 'both'")

# --- Ví dụ dùng ---
# img là mảng numpy 2D (grayscale)
img_np = load_image_from_path("training_images\maysay_segment_0127.png")
img_np = flip_image(img_np,'both')
img_out, darkness, max_corner = flip_to_max_top_right(img_np)
# print(darkness, "max at:", max_corner)


# Hiển thị ảnh trước và sau lật
fig, axs = plt.subplots(1, 2, figsize=(6, 3))
axs[0].imshow(img_np, cmap='gray')
axs[0].set_title("Before Flip")
axs[1].imshow(img_out, cmap='gray')
axs[1].set_title("After Flip")
plt.show()
