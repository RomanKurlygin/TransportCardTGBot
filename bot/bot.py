import os
import asyncio
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, CallbackQuery, FSInputFile
from aiogram.filters import CommandStart
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton

from cv.card_ocr import extract_card_number
from web.balance_site import create_balance_session_and_get_captcha, submit_captcha_and_get_result
from training.captcha_dataset import save_labeled_captcha
from cv.captcha_solver import predict_captcha
from cv.result_parser import parse_balance_result, format_balance_result

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")

if not BOT_TOKEN:
    raise ValueError("Не найден BOT_TOKEN в файле .env")


DOWNLOAD_DIR = Path("../downloads")
LOG_DIR = Path("../logs")

DOWNLOAD_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)


bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()


# Временное хранилище данных пользователя
user_data = {}


def write_log(
    image_path: str,
    card_number: str | None,
    raw_text: str,
    status: str
):
    """
    Записывает результат распознавания в logs/log.txt
    """

    log_path = LOG_DIR / "log.txt"

    time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(log_path, "a", encoding="utf-8") as file:
        file.write("====================================\n")
        file.write(f"Время: {time_now}\n")
        file.write(f"Файл: {image_path}\n")
        file.write(f"Номер карты: {card_number}\n")
        file.write(f"OCR текст: {raw_text.strip()}\n")
        file.write(f"Статус: {status}\n")


def get_confirm_keyboard():
    """
    Кнопки подтверждения номера карты
    """

    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="Да, номер верный",
                    callback_data="card_number_confirm"
                )
            ],
            [
                InlineKeyboardButton(
                    text="Исправить номер",
                    callback_data="card_number_edit"
                )
            ]
        ]
    )

    return keyboard

def get_captcha_confirm_keyboard():
    keyboard = InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text="Да, captcha верная",
                    callback_data="captcha_confirm"
                )
            ],
            [
                InlineKeyboardButton(
                    text="Исправить captcha",
                    callback_data="captcha_edit"
                )
            ]
        ]
    )

    return keyboard


@dp.message(CommandStart())
async def start_handler(message: Message):
    await message.answer(
        "Привет! Отправь мне фото транспортной карты, "
        "и я попробую распознать её номер."
    )


@dp.message(F.photo)
async def photo_handler(message: Message):
    await message.answer("Фото получено. Считываю номер карты...")

    photo = message.photo[-1]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = DOWNLOAD_DIR / f"card_{message.from_user.id}_{timestamp}.jpg"

    await bot.download(photo, destination=file_path)

    try:
        card_number, raw_text = extract_card_number(str(file_path))

        user_id = message.from_user.id

        user_data[user_id] = {
            "image_path": str(file_path),
            "card_number": card_number,
            "raw_text": raw_text,
            "waiting_for_card_edit": False
        }

        if card_number:
            write_log(
                image_path=str(file_path),
                card_number=card_number,
                raw_text=raw_text,
                status="ocr_detected_waiting_confirmation"
            )

            await message.answer(
                f"Номер карты распознан:\n\n"
                f"`{card_number}`\n\n"
                f"Проверь, правильно ли распознан номер.",
                parse_mode="Markdown",
                reply_markup=get_confirm_keyboard()
            )
        else:
            write_log(
                image_path=str(file_path),
                card_number=None,
                raw_text=raw_text,
                status="ocr_failed"
            )

            await message.answer(
                "Не удалось уверенно распознать номер карты.\n\n"
                "Введи номер карты вручную сообщением."
            )

            user_data[user_id]["waiting_for_card_edit"] = True

    except Exception as e:
        await message.answer(f"Ошибка при обработке изображения:\n{e}")


@dp.callback_query(F.data == "card_number_confirm")
async def confirm_card_number(callback: CallbackQuery):
    user_id = callback.from_user.id

    if user_id not in user_data:
        await callback.message.answer("Сначала отправь фото карты.")
        await callback.answer()
        return

    card_number = user_data[user_id].get("card_number")
    image_path = user_data[user_id].get("image_path")
    raw_text = user_data[user_id].get("raw_text", "")

    write_log(
        image_path=image_path,
        card_number=card_number,
        raw_text=raw_text,
        status="card_number_confirmed"
    )

    await callback.message.edit_text(
        f"Номер карты подтверждён:\n\n"
        f"`{card_number}`\n\n"
        f"Открываю сайт, ввожу номер карты и получаю captcha...",
        parse_mode="Markdown"
    )

    try:
        session, captcha_path = await create_balance_session_and_get_captcha(card_number)

        user_data[user_id]["balance_session"] = session
        user_data[user_id]["captcha_path"] = captcha_path

        captcha_file = FSInputFile(captcha_path)

        captcha_pred = predict_captcha(captcha_path)

        user_data[user_id]["captcha_pred"] = captcha_pred
        user_data[user_id]["waiting_for_captcha_text"] = False

        await callback.message.answer_photo(
            photo=captcha_file,
            caption=(
                f"Captcha получена с сайта.\n\n"
                f"Модель CNN распознала captcha как:\n"
                f"`{captcha_pred}`\n\n"
                f"Проверь, правильно ли распознан код."
            ),
            parse_mode="Markdown",
            reply_markup=get_captcha_confirm_keyboard()
        )

        write_log(
            image_path=image_path,
            card_number=card_number,
            raw_text=f"captcha_path={captcha_path}",
            status="captcha_received"
        )

    except Exception as e:
        await callback.message.answer(
            f"Ошибка при получении captcha:\n{e}"
        )

    await callback.answer()


@dp.callback_query(F.data == "card_number_edit")
async def edit_card_number(callback: CallbackQuery):
    user_id = callback.from_user.id

    if user_id not in user_data:
        await callback.message.answer("Сначала отправь фото карты.")
        await callback.answer()
        return

    user_data[user_id]["waiting_for_card_edit"] = True

    await callback.message.edit_text(
        "Хорошо. Введи правильный номер карты сообщением.\n\n"
        "Например:\n"
        "`3715354948`",
        parse_mode="Markdown"
    )

    await callback.answer()


async def submit_captcha_to_site_and_answer(message_or_callback, user_id: int, captcha_text: str):
    """
    Вводит captcha на сайте, нажимает кнопку выполнения запроса,
    парсит результат и отправляет красивое сообщение пользователю.
    """

    session = user_data[user_id].get("balance_session")

    if not session:
        await message_or_callback.answer(
            "Сессия сайта не найдена. Нужно заново отправить фото карты."
        )
        return

    try:
        result_text = await submit_captcha_and_get_result(session, captcha_text)

        await session.close()
        user_data[user_id]["balance_session"] = None

        parsed_result = parse_balance_result(result_text)
        formatted_result = format_balance_result(parsed_result)

        await message_or_callback.answer(
            formatted_result,
            parse_mode="Markdown"
        )

        write_log(
            image_path=user_data[user_id].get("image_path", "unknown"),
            card_number=user_data[user_id].get("card_number"),
            raw_text=f"captcha={captcha_text}; parsed_result={parsed_result}",
            status="balance_request_submitted"
        )

    except Exception as e:
        await message_or_callback.answer(
            f"Ошибка при отправке captcha на сайт:\n{e}"
        )

        try:
            await session.close()
        except Exception:
            pass



@dp.callback_query(F.data == "captcha_confirm")
async def confirm_captcha(callback: CallbackQuery):
    user_id = callback.from_user.id

    if user_id not in user_data:
        await callback.message.answer("Сначала получи captcha.")
        await callback.answer()
        return

    captcha_text = user_data[user_id].get("captcha_pred")
    captcha_path = user_data[user_id].get("captcha_path")
    card_number = user_data[user_id].get("card_number")
    image_path = user_data[user_id].get("image_path", "unknown")

    if not captcha_text:
        await callback.message.answer("Не найдено предсказание captcha.")
        await callback.answer()
        return

    try:
        saved_path = save_labeled_captcha(captcha_path, captcha_text)

        user_data[user_id]["captcha_text"] = captcha_text

        write_log(
            image_path=image_path,
            card_number=card_number,
            raw_text=f"captcha_pred={captcha_text}; captcha_saved={saved_path}",
            status="captcha_confirmed_and_saved"
        )

        await callback.message.edit_caption(
            caption=(
                f"Captcha подтверждена:\n\n"
                f"`{captcha_text}`\n\n"
                f"Файл сохранён в датасет:\n"
                f"`{saved_path}`\n\n"
                f"Отправляю captcha на сайт..."
            ),
            parse_mode="Markdown"
        )

        await submit_captcha_to_site_and_answer(
            callback.message,
            user_id,
            captcha_text
        )

    except Exception as e:
        await callback.message.answer(f"Ошибка при сохранении captcha:\n{e}")

    await callback.answer()

@dp.callback_query(F.data == "captcha_edit")
async def edit_captcha(callback: CallbackQuery):
    user_id = callback.from_user.id

    if user_id not in user_data:
        await callback.message.answer("Сначала получи captcha.")
        await callback.answer()
        return

    user_data[user_id]["waiting_for_captcha_text"] = True

    await callback.message.answer(
        "Введи правильный код captcha сообщением.\n\n"
        "Например:\n"
        "`2137`",
        parse_mode="Markdown"
    )

    await callback.answer()




@dp.message(F.text)
async def text_handler(message: Message):
    user_id = message.from_user.id
    text = message.text.strip()

    if user_id in user_data and user_data[user_id].get("waiting_for_card_edit"):
        if not text.isdigit():
            await message.answer(
                "Номер карты должен содержать только цифры. "
                "Попробуй ещё раз."
            )
            return

        if len(text) < 6:
            await message.answer(
                "Номер слишком короткий. Введи номер карты ещё раз."
            )
            return

        user_data[user_id]["card_number"] = text
        user_data[user_id]["waiting_for_card_edit"] = False

        image_path = user_data[user_id].get("image_path", "manual_input")
        raw_text = user_data[user_id].get("raw_text", "")

        write_log(
            image_path=image_path,
            card_number=text,
            raw_text=raw_text,
            status="card_number_corrected_manually"
        )

        await message.answer(
            f"Номер карты сохранён:\n\n"
            f"`{text}`\n\n"
            f"Проверь, правильно ли указан номер.",
            parse_mode="Markdown",
            reply_markup=get_confirm_keyboard()
        )

        return

    if user_id in user_data and user_data[user_id].get("waiting_for_captcha_text"):
        if not text.isdigit():
            await message.answer(
                "Код captcha должен содержать только цифры. "
                "Попробуй ещё раз."
            )
            return

        if len(text) != 4:
            await message.answer(
                "Код captcha должен состоять ровно из 4 цифр. "
                "Попробуй ещё раз."
            )
            return

        captcha_path = user_data[user_id].get("captcha_path")
        card_number = user_data[user_id].get("card_number")
        image_path = user_data[user_id].get("image_path", "unknown")

        try:
            saved_path = save_labeled_captcha(captcha_path, text)

            user_data[user_id]["captcha_text"] = text
            user_data[user_id]["waiting_for_captcha_text"] = False

            write_log(
                image_path=image_path,
                card_number=card_number,
                raw_text=f"captcha_text={text}; captcha_saved={saved_path}",
                status="captcha_labeled_saved"
            )

            await message.answer(
                f"Captcha сохранена в датасет:\n\n"
                f"`{saved_path}`\n\n"
                f"Код captcha: `{text}`",
                parse_mode="Markdown"
            )

            await message.answer(
                "Captcha сохранена. Отправляю код на сайт..."
            )

            await submit_captcha_to_site_and_answer(
                message,
                user_id,
                text
            )

        except Exception as e:
            await message.answer(f"Ошибка при сохранении captcha:\n{e}")

        return

    await message.answer("Пожалуйста, отправь фото транспортной карты.")






async def main():
    print("Бот запущен...")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())