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
                    "p_tolerance": float(row["p_tolerance"])
                })

    def match(self, label, P_mean):
        """
        Trả về label nếu khớp, "null" nếu không khớp.
        """
        for tpl in self.templates:
            # 1. Kiểm tra nhãn
            if tpl["label"] != label:
                continue

            # 2. So sánh P_mean
            p_similarity = min(P_mean, tpl["P_mean"]) / max(P_mean, tpl["P_mean"])
            print(f"P_similarity = {p_similarity:.4f}")

            if p_similarity >= tpl["p_tolerance"]:
                return tpl["label"] 
            
        return "null"
