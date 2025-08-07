import torch
from torchvision import transforms
from PIL import Image
import joblib
from MLP import MLP  
import numpy as np

class MLP_Predict:
    def __init__(self, model_path, label_encoder_path, image_size=32):
        self.image_size = image_size
        self.img_input_size = image_size * image_size
        self.p_input_size = 1

        # Load label encoder
        self.label_encoder = joblib.load(label_encoder_path)
        self.num_classes = len(self.label_encoder.classes_)

        # Load model
        self.model = MLP(self.img_input_size, self.p_input_size, self.num_classes)
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()

        # Transform ảnh
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

    def predict(self, image_input, p_mean):
        """
        image_input: str (đường dẫn ảnh), numpy.ndarray, hoặc PIL.Image.Image
        p_mean: float

        Returns:
            predicted_label (str): Nhãn dự đoán
            confidence (float): Xác suất tương ứng (0–1)
        """
        # Chuẩn hóa ảnh đầu vào
        if isinstance(image_input, str):
            img = Image.open(image_input).convert("L")
            img_tensor = self.transform(img).view(1, -1)
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:
                img = Image.fromarray(image_input.astype(np.uint8), mode='L')
            elif image_input.ndim == 3 and image_input.shape[2] == 1:
                img = Image.fromarray(image_input.squeeze().astype(np.uint8), mode='L')
            else:
                raise ValueError("numpy.ndarray phải là ảnh grayscale 2D (HxW)")
            img_tensor = self.transform(img).view(1, -1)
        elif isinstance(image_input, Image.Image):
            img = image_input.convert("L")
            img_tensor = self.transform(img).view(1, -1)
        else:
            raise ValueError("image_input phải là đường dẫn (str), numpy.ndarray hoặc PIL.Image.Image")

        # Chuẩn hóa p_mean
        p_tensor = torch.tensor([[p_mean]], dtype=torch.float32)

        # Dự đoán
        with torch.no_grad():
            output = self.model(img_tensor, p_tensor)  # (1, num_classes)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            predicted_label = self.label_encoder.inverse_transform([predicted_class])[0]
            confidence = probabilities[0, predicted_class].item()
        return predicted_label, confidence




