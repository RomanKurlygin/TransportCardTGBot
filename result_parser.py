import re


def clean_text(text: str) -> str:
    """
    Убираем лишние пробелы и приводим текст к удобному виду.
    """
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n+", "\n", text)
    return text.strip()


def extract_between(text: str, start_label: str, end_labels: list[str]) -> str | None:
    """
    Извлекает значение после start_label до ближайшей следующей метки.
    """

    pattern_start = re.escape(start_label)

    end_pattern = "|".join(re.escape(label) for label in end_labels)

    pattern = rf"{pattern_start}\s*:?\s*(.*?)(?={end_pattern}|$)"

    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)

    if not match:
        return None

    value = match.group(1).strip()

    # Чистим переносы и лишние пробелы
    value = re.sub(r"\s+", " ", value).strip()

    return value if value else None


def parse_balance_result(text: str) -> dict:
    """
    Парсит текст страницы результата проверки баланса.
    """

    text = clean_text(text)

    # Берём только часть текста, начиная с результата проверки карты
    card_start = re.search(r"Карта\s*№\s*\d+", text)

    if card_start:
        text = text[card_start.start():]

    # Отрезаем футер сайта, чтобы он не попадал в поля
    footer_markers = [
        "Виды карт",
        "Услуги",
        "О компании",
        "Где приобрести",
        "Где пополнить",
        "Контакты",
        "Вопросы и ответы",
        "АО «Транспортная карта»"
    ]

    for marker in footer_markers:
        pos = text.find(marker)
        if pos != -1:
            text = text[:pos]
            break

    labels = [
        "Проездной билет",
        "Действует",
        "Ресурс сейчас",
        "Последнее предъявление в транспорте",
        "Маршрут",
        "Вид транспорта",
        "Операция",
        "Дата и время пополнения",
        "Пункт пополнения",
        "Ресурс пополнен на",
    ]

    result = {}

    card_match = re.search(r"Карта\s*№\s*(\d+)", text)
    result["card_number"] = card_match.group(1) if card_match else None

    result["ticket_type"] = extract_between(
        text,
        "Проездной билет",
        labels[1:]
    )

    result["valid_until"] = extract_between(
        text,
        "Действует",
        labels[2:]
    )

    result["balance"] = extract_between(
        text,
        "Ресурс сейчас",
        labels[3:]
    )

    result["last_use"] = extract_between(
        text,
        "Последнее предъявление в транспорте",
        labels[4:]
    )

    result["route"] = extract_between(
        text,
        "Маршрут",
        labels[5:]
    )

    result["transport_type"] = extract_between(
        text,
        "Вид транспорта",
        labels[6:]
    )

    result["operation"] = extract_between(
        text,
        "Операция",
        labels[7:]
    )

    result["topup_time"] = extract_between(
        text,
        "Дата и время пополнения",
        labels[8:]
    )

    result["topup_point"] = extract_between(
        text,
        "Пункт пополнения",
        labels[9:]
    )

    result["topup_amount"] = extract_between(
        text,
        "Ресурс пополнен на",
        footer_markers
    )

    return result


def format_balance_result(data: dict) -> str:
    """
    Формирует красивое сообщение для Telegram.
    """

    def value(key):
        return data.get(key) or "не найдено"

    message = (
        "Результат проверки карты:\n\n"
        f"Номер карты: `{value('card_number')}`\n"
        f"Тип билета: {value('ticket_type')}\n"
        f"Срок действия: {value('valid_until')}\n"
        f"Баланс: *{value('balance')}*\n"
        f"Последнее предъявление: {value('last_use')}\n"
        f"Маршрут: {value('route')}\n"
        f"Вид транспорта: {value('transport_type')}\n"
        f"Операция: {value('operation')}\n"
        f"Дата пополнения: {value('topup_time')}\n"
        f"Пункт пополнения: {value('topup_point')}\n"
        f"Пополнено на: {value('topup_amount')}"
    )

    return message