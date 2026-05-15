import re
from pathlib import Path

import cv2
import easyocr


DEBUG_DIR = Path("debug")
DEBUG_DIR.mkdir(exist_ok=True)

# Создаём OCR-reader один раз
reader = easyocr.Reader(["en"], gpu=False)


def crop_card_number_area(image):
    """
    Вырезаем область номера карты.
    Для данной карты номер находится справа снизу.
    """

    h, w = image.shape[:2]

    x1 = int(w * 0.55)
    y1 = int(h * 0.60)
    x2 = int(w * 0.99)
    y2 = int(h * 0.90)

    crop = image[y1:y2, x1:x2]

    return crop


def preprocess_for_easyocr(crop):
    """
    Для EasyOCR не нужно сильно портить изображение бинаризацией.
    Лучше оставить серое или контрастное изображение.
    """

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # Увеличиваем изображение
    gray_big = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

    # Немного повышаем контраст
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(gray_big)

    return contrast


def extract_digits(text):
    return re.sub(r"\D", "", text)


def extract_card_number(image_path: str):
    """
    Распознаёт номер транспортной карты с изображения.
    Возвращает:
    - номер карты
    - сырой распознанный текст
    """

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Не удалось открыть изображение")

    crop = crop_card_number_area(image)
    processed = preprocess_for_easyocr(crop)

    cv2.imwrite(str(DEBUG_DIR / "crop_number.jpg"), crop)
    cv2.imwrite(str(DEBUG_DIR / "easyocr_processed.jpg"), processed)

    # EasyOCR
    results = reader.readtext(
        processed,
        detail=1,
        allowlist="0123456789"
    )

    all_texts = []

    for bbox, text, confidence in results:
        digits = extract_digits(text)

        all_texts.append({
            "text": text,
            "digits": digits,
            "confidence": confidence
        })

    # Сохраняем отладку
    with open(DEBUG_DIR / "easyocr_results.txt", "w", encoding="utf-8") as f:
        for item in all_texts:
            f.write(f"text: {item['text']}\n")
            f.write(f"digits: {item['digits']}\n")
            f.write(f"confidence: {item['confidence']}\n")
            f.write("-" * 40 + "\n")

    candidates = []

    for item in all_texts:
        digits = item["digits"]

        # У твоей карты номер 10 цифр
        found = re.findall(r"\d{10}", digits)
        for number in found:
            candidates.append((number, item["confidence"]))

    if candidates:
        # Берём вариант с максимальной уверенностью
        best_number = max(candidates, key=lambda x: x[1])[0]
        return best_number, best_number

    # Если EasyOCR разбил номер на части, склеиваем все цифры
    joined_digits = "".join(item["digits"] for item in all_texts)

    found = re.findall(r"\d{8,12}", joined_digits)

    if found:
        best_number = max(found, key=len)
        return best_number, joined_digits

    return None, joined_digits