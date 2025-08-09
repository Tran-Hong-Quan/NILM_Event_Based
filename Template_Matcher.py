import os
import csv
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

class TemplateMatcher:
    def __init__(self, dataset_dir):
        """
        dataset_dir: thư mục chứa metadata.csv và ảnh mẫu
        """
        self.dataset_dir = dataset_dir
        self.templates = []
        self._load_templates()

    def _load_templates(self):
        csv_path = os.path.join(self.dataset_dir, "metadata.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"metadata.csv không tồn tại trong {self.dataset_dir}")

        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                template_path = os.path.join(self.dataset_dir, row["filename"])
                if not os.path.exists(template_path):
                    continue
                self.templates.append({
                    "filename": template_path,
                    "P_mean": float(row["P_mean"]),
                    "label": row["label"],
                    "p_tolerance": float(row["p_tolerance"]),
                    "img_tolerance": float(row["img_tolerance"])
                })

    def _load_image(self, img):
        """
        Chuyển ảnh đầu vào thành numpy grayscale 0-1
        """
        if isinstance(img, np.ndarray):
            arr = img
        elif isinstance(img, Image.Image):
            arr = np.array(img)
        elif isinstance(img, str):
            arr = np.array(Image.open(img))
        else:
            raise TypeError("Ảnh phải là numpy.ndarray, PIL.Image.Image hoặc đường dẫn (str)")

        # Chuyển sang grayscale nếu là ảnh màu
        if arr.ndim == 3:
            arr = np.mean(arr, axis=2)
        arr = arr.astype(np.float32) / 255.0
        return arr

    def match(self, label, P_mean, img):
        """
        Trả về label nếu khớp, "null" nếu không khớp
        """
        img_arr = self._load_image(img)

        for tpl in self.templates:
            # 1. Check nhãn
            if tpl["label"] != label:
                continue

            p_similarity = min(P_mean, tpl["P_mean"]) / max(P_mean, tpl["P_mean"])
            if p_similarity < tpl["p_tolerance"]:  # ví dụ: p_tolerance = 0.95
                continue

            # 3. Check ảnh tolerance
            tpl_img = self._load_image(tpl["filename"])
            # Resize ảnh đầu vào về kích thước mẫu để so sánh
            if tpl_img.shape != img_arr.shape:
                img_arr_resized = np.array(Image.fromarray((img_arr * 255).astype(np.uint8)).resize(
                    (tpl_img.shape[1], tpl_img.shape[0]), Image.BILINEAR)) / 255.0
            else:
                img_arr_resized = img_arr

            score = ssim(tpl_img, img_arr_resized, data_range=1.0)
            diff = 1 - score  # độ khác biệt (0 là giống hệt)
            print("Img diff = " + str(diff))
            print("P_Sim = "+str(p_similarity))
            if diff <= tpl["img_tolerance"]:
                return tpl["label"]

        return "null"
