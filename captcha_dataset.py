from pathlib import Path
from datetime import datetime
import shutil


CAPTCHA_DATASET_DIR = Path("dataset/captcha/labeled")
CAPTCHA_DATASET_DIR.mkdir(parents=True, exist_ok=True)


def save_labeled_captcha(captcha_path: str, captcha_text: str) -> str:
    """
    Сохраняет captcha в датасет с правильной меткой.

    Пример имени файла:
    3127_20260515_115530_123456.png
    """

    captcha_text = captcha_text.strip()

    if not captcha_text.isdigit():
        raise ValueError("Captcha должна содержать только цифры")

    if len(captcha_text) != 4:
        raise ValueError("Captcha должна состоять ровно из 4 цифр")

    source_path = Path(captcha_path)

    if not source_path.exists():
        raise FileNotFoundError(f"Файл captcha не найден: {captcha_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    new_filename = f"{captcha_text}_{timestamp}.png"
    new_path = CAPTCHA_DATASET_DIR / new_filename

    shutil.copy2(source_path, new_path)

    return str(new_path)