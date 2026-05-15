import re
from pathlib import Path

import cv2
import easyocr


DEBUG_DIR = Path("debug")
DEBUG_DIR.mkdir(exist_ok=True)

reader = easyocr.Reader(["en"], gpu=False)


def extract_digits(text: str) -> str:
    return re.sub(r"\D", "", text)


def preprocess_full_image(image):
    """
    Предобработка всего изображения для EasyOCR.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Увеличиваем фото, чтобы цифры были крупнее
    h, w = gray.shape[:2]

    if max(h, w) < 1200:
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Немного повышаем контраст
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(gray)

    return contrast


def crop_card_number_area(image):
    """
    Старый запасной вариант: вырезаем правую нижнюю область.
    """
    h, w = image.shape[:2]

    x1 = int(w * 0.55)
    y1 = int(h * 0.60)
    x2 = int(w * 0.99)
    y2 = int(h * 0.90)

    return image[y1:y2, x1:x2]


def preprocess_crop(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray_big = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(gray_big)

    return contrast


def find_10_digit_number_from_easyocr(results):
    """
    Ищем 10-значный номер среди результатов EasyOCR.
    Возвращаем самый уверенный вариант.
    """

    candidates = []

    for item in results:
        # EasyOCR возвращает: bbox, text, confidence
        bbox, text, confidence = item

        digits = extract_digits(text)

        found_numbers = re.findall(r"\d{10}", digits)

        for number in found_numbers:
            candidates.append((number, confidence, text))

    if not candidates:
        return None, ""

    # Берём кандидата с максимальной confidence
    best_number, best_confidence, best_raw_text = max(
        candidates,
        key=lambda x: x[1]
    )

    return best_number, best_raw_text


def extract_card_number(image_path: str):
    """
    Распознаёт 10-значный номер транспортной карты.
    Сначала ищет номер на всём изображении.
    Если не получилось — пробует старый crop-метод.
    """

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Не удалось открыть изображение")

    # ---------- 1. OCR по всему изображению ----------
    full_processed = preprocess_full_image(image)
    cv2.imwrite(str(DEBUG_DIR / "full_processed.jpg"), full_processed)

    full_results = reader.readtext(
        full_processed,
        detail=1,
        allowlist="0123456789"
    )

    number, raw_text = find_10_digit_number_from_easyocr(full_results)

    with open(DEBUG_DIR / "easyocr_full_results.txt", "w", encoding="utf-8") as f:
        for bbox, text, confidence in full_results:
            f.write(f"text: {text}\n")
            f.write(f"digits: {extract_digits(text)}\n")
            f.write(f"confidence: {confidence}\n")
            f.write("-" * 40 + "\n")

    if number:
        return number, raw_text

    # ---------- 2. Если на всём фото не нашли — fallback crop ----------
    crop = crop_card_number_area(image)
    crop_processed = preprocess_crop(crop)

    cv2.imwrite(str(DEBUG_DIR / "crop_number.jpg"), crop)
    cv2.imwrite(str(DEBUG_DIR / "crop_processed.jpg"), crop_processed)

    crop_results = reader.readtext(
        crop_processed,
        detail=1,
        allowlist="0123456789"
    )

    number, raw_text = find_10_digit_number_from_easyocr(crop_results)

    with open(DEBUG_DIR / "easyocr_crop_results.txt", "w", encoding="utf-8") as f:
        for bbox, text, confidence in crop_results:
            f.write(f"text: {text}\n")
            f.write(f"digits: {extract_digits(text)}\n")
            f.write(f"confidence: {confidence}\n")
            f.write("-" * 40 + "\n")

    if number:
        return number, raw_text

    return None, ""