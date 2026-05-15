from pathlib import Path
import random
from datetime import datetime

from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np


OUTPUT_DIR = Path("dataset/captcha/synthetic")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WIDTH = 201
HEIGHT = 61


def find_font():
    """
    Пробуем найти стандартный жирный шрифт Windows.
    Если не нашли — используем дефолтный PIL.
    """

    possible_fonts = [
        r"C:\Windows\Fonts\arialbd.ttf",
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\calibrib.ttf",
        r"C:\Windows\Fonts\calibri.ttf",
        r"C:\Windows\Fonts\timesbd.ttf",
    ]

    for font_path in possible_fonts:
        if Path(font_path).exists():
            return font_path

    return None


FONT_PATH = find_font()


def add_noise(image: Image.Image, amount: int = 900):
    """
    Добавляет шум в виде чёрных точек, похожий на captcha сайта.
    """

    draw = ImageDraw.Draw(image)

    for _ in range(amount):
        x = random.randint(0, WIDTH - 1)
        y = random.randint(0, HEIGHT - 1)

        color = random.choice([
            (0, 0, 0),
            (30, 30, 30),
            (80, 80, 80)
        ])

        radius = random.choice([1, 1, 1, 2])

        draw.ellipse(
            (x, y, x + radius, y + radius),
            fill=color
        )

    return image


def create_captcha(text: str):
    """
    Генерирует одну captcha по стилю, похожему на сайт:
    - 4 цифры
    - белый фон
    - чёрные крупные цифры
    - шум точками
    - небольшие повороты символов
    """

    image = Image.new("RGB", (WIDTH, HEIGHT), "white")

    # Сначала шум на фон
    image = add_noise(image, amount=random.randint(700, 1200))

    # Отдельно рисуем цифры
    digit_layer = Image.new("RGBA", (WIDTH, HEIGHT), (255, 255, 255, 0))

    x_positions = [18, 62, 107, 153]

    for i, digit in enumerate(text):
        digit_img = Image.new("RGBA", (50, 60), (255, 255, 255, 0))
        digit_draw = ImageDraw.Draw(digit_img)

        font_size = random.randint(38, 46)

        if FONT_PATH:
            font = ImageFont.truetype(FONT_PATH, font_size)
        else:
            font = ImageFont.load_default()

        # Небольшой разброс по положению
        dx = random.randint(-3, 3)
        dy = random.randint(-5, 3)

        digit_draw.text(
            (8 + dx, 4 + dy),
            digit,
            font=font,
            fill=(0, 0, 0, 255)
        )

        # Небольшой поворот цифры
        angle = random.uniform(-8, 8)
        digit_img = digit_img.rotate(
            angle,
            resample=Image.BICUBIC,
            expand=False
        )

        digit_layer.alpha_composite(
            digit_img,
            (x_positions[i] + random.randint(-3, 3), random.randint(0, 4))
        )

    image = Image.alpha_composite(image.convert("RGBA"), digit_layer)

    # Лёгкое размытие, как от скриншота/сжатия
    if random.random() < 0.4:
        image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.6)))

    # Иногда немного меняем яркость
    arr = np.array(image.convert("RGB")).astype(np.int16)
    shift = random.randint(-10, 10)
    arr = np.clip(arr + shift, 0, 255).astype(np.uint8)

    image = Image.fromarray(arr)

    return image


def generate_dataset(count: int):
    for i in range(count):
        text = "".join(str(random.randint(0, 9)) for _ in range(4))

        image = create_captcha(text)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{text}_synthetic_{timestamp}_{i:04d}.png"

        image.save(OUTPUT_DIR / filename)

    print(f"Сгенерировано synthetic captcha: {count}")
    print(f"Папка: {OUTPUT_DIR}")


if __name__ == "__main__":
    count = int(input("Сколько synthetic captcha сгенерировать? Например 500: "))
    generate_dataset(count)