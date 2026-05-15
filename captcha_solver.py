from pathlib import Path

from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as transforms


MODEL_PATH = Path("models/captcha_cnn.pth")


class CaptchaCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 20, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.head1 = nn.Linear(256, 10)
        self.head2 = nn.Linear(256, 10)
        self.head3 = nn.Linear(256, 10)
        self.head4 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        d1 = self.head1(x)
        d2 = self.head2(x)
        d3 = self.head3(x)
        d4 = self.head4(x)

        return d1, d2, d3, d4


def load_captcha_model(device=None):
    """
    Загружает обученную CNN-модель captcha.
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Файл модели не найден: {MODEL_PATH}. "
            f"Сначала запусти train_captcha.py"
        )

    model = CaptchaCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    return model, device


def predict_captcha(image_path: str, model=None, device=None):
    """
    Распознаёт captcha по изображению.
    Возвращает строку из 4 цифр.
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Файл captcha не найден: {image_path}")

    if model is None:
        model, device = load_captcha_model(device)

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((50, 160)),
        transforms.ToTensor()
    ])

    image = Image.open(image_path).convert("RGB")
    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        d1, d2, d3, d4 = model(x)

    pred_digits = [
        d1.argmax(dim=1).item(),
        d2.argmax(dim=1).item(),
        d3.argmax(dim=1).item(),
        d4.argmax(dim=1).item()
    ]

    captcha_text = "".join(map(str, pred_digits))

    return captcha_text


if __name__ == "__main__":
    test_path = "debug/site_captcha.png"

    model, device = load_captcha_model()
    result = predict_captcha(test_path, model, device)

    print(f"Captcha image: {test_path}")
    print(f"Predicted captcha: {result}")