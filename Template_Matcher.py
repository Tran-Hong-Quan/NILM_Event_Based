import os
import csv
import json
import numpy as np
from Utilis.NILM_Utilis import align_phase

class TemplateMatcher:
    def __init__(self, dataset_dir):
        """
        dataset_dir: thư mục chứa Template_data.csv
        """
        self.dataset_dir = dataset_dir
        self.templates = []
        self._load_templates()

    def _load_templates(self):
        csv_path = os.path.join(self.dataset_dir, "Template_data.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Template_data.csv không tồn tại trong {self.dataset_dir}")

        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.templates.append({
                    "P_mean": float(row["P_mean"]),
                    "label": row["label"],
                    "p_tolerance": float(row["p_tolerance"]),
                    "ui_tolerance": float(row["ui_tolerance"]),
                    "U_array": np.array(json.loads(row["U_array"])),
                    "I_array": np.array(json.loads(row["I_array"]))
                })

    def match(self, label, P_mean, U_input, I_input):
        """
        Trả về label nếu khớp, "null" nếu không khớp.
        U_input, I_input: numpy array
        """
        for tpl in self.templates:
            # 1. Kiểm tra nhãn
            if tpl["label"] != label:
                continue

            # 2. So sánh P_mean
            p_similarity = min(P_mean, tpl["P_mean"]) / max(P_mean, tpl["P_mean"])
            print(f"P_similarity = {p_similarity:.4f}")

            if p_similarity < tpl["p_tolerance"]:
                continue

            # 3. So sánh U/I sau khi căn pha
            U_tpl = tpl["U_array"]
            I_tpl = tpl["I_array"]

            U_tpl_aligned, best_shift = align_phase(U_input, U_tpl)
            I_tpl_aligned = np.roll(I_tpl, -int(best_shift))

            # Sai số trung bình tuyệt đối
            U_error = np.mean(np.abs(U_input - U_tpl_aligned))
            I_error = np.mean(np.abs(I_input - I_tpl_aligned))

            print(f"U_error = {U_error:.4f}, I_error = {I_error:.4f}")

            # Công suất sai số
            P_error = U_error * I_error

            # Công suất template (RMS-based))
            P_tpl =  tpl["P_mean"]

            # Độ giống nhau dựa trên công suất
            p_similarity_error = min(P_error, P_tpl) / max(P_error, P_tpl)
            p_similarity_error = 1-p_similarity_error
            print(f"P_error = {P_error:.4f}, P_tpl = {P_tpl:.4f}, similarity = {p_similarity_error:.4f}")

            if p_similarity_error >= tpl["ui_tolerance"]:
                return tpl["label"]


        return "null"
