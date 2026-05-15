import asyncio
from pathlib import Path

from balance_site import open_balance_page_accept_and_get_captcha
from captcha_dataset import save_labeled_captcha


def count_labeled_captchas() -> int:
    dataset_dir = Path("dataset/captcha/labeled")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    return len(list(dataset_dir.glob("*.png")))


async def collect_one_captcha():
    print("=" * 50)
    print("Получаю captcha с сайта...")

    captcha_path = await open_balance_page_accept_and_get_captcha()

    print(f"Captcha сохранена во временный файл: {captcha_path}")
    print("Открой изображение и введи код с картинки.")
    print("Для отмены введи: q")

    while True:
        code = input("Код captcha: ").strip()

        if code.lower() == "q":
            print("Сохранение отменено.")
            return

        if not code.isdigit():
            print("Ошибка: код должен содержать только цифры.")
            continue

        if len(code) != 4:
            print("Ошибка: код должен состоять ровно из 4 цифр.")
            continue

        saved_path = save_labeled_captcha(captcha_path, code)

        print(f"Captcha сохранена в датасет: {saved_path}")
        print(f"Всего captcha в датасете: {count_labeled_captchas()}")
        return


async def main():
    print("Сбор датасета captcha")
    print("Фото карты и номер карты для этого этапа не нужны.")
    print()

    try:
        n = int(input("Сколько captcha собрать? Например 10: ").strip())
    except ValueError:
        print("Нужно ввести число.")
        return

    for i in range(n):
        print()
        print(f"Captcha {i + 1} из {n}")

        try:
            await collect_one_captcha()
        except Exception as e:
            print(f"Ошибка при сборе captcha: {e}")

    print()
    print("Сбор завершён.")
    print(f"Всего captcha в датасете: {count_labeled_captchas()}")


if __name__ == "__main__":
    asyncio.run(main())