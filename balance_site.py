from pathlib import Path
from playwright.async_api import async_playwright


DEBUG_DIR = Path("debug")
DEBUG_DIR.mkdir(exist_ok=True)

BALANCE_URL = "https://transkart.ru/services/check-balance"


async def click_cookie_accept(page):
    selectors = [
        'button:has-text("Принять")',
        'text=Принять'
    ]

    for selector in selectors:
        try:
            locator = page.locator(selector).first
            if await locator.count() > 0:
                await locator.click(timeout=3000)
                print("Cookie-окно закрыто")
                return True
        except Exception:
            pass

    print("Cookie-окно не найдено или уже закрыто")
    return False


async def click_accept_terms(page):
    selectors = [
        'input[type="button"][value*="Я согласен"]',
        'input[type="submit"][value*="Я согласен"]',
        'button:has-text("Я согласен")',
        'text=Я согласен',
        'text=Я согласен(на)'
    ]

    for selector in selectors:
        try:
            locator = page.locator(selector).first
            if await locator.count() > 0:
                await locator.scroll_into_view_if_needed(timeout=3000)
                await locator.click(timeout=5000)
                print(f"Кнопка согласия нажата на основной странице: {selector}")
                return True
        except Exception:
            pass

    for frame in page.frames:
        for selector in selectors:
            try:
                locator = frame.locator(selector).first
                if await locator.count() > 0:
                    await locator.scroll_into_view_if_needed(timeout=3000)
                    await locator.click(timeout=5000)
                    print(f"Кнопка согласия нажата во frame: {selector}")
                    return True
            except Exception:
                pass

    return False


async def fill_card_number(page, card_number: str):
    selectors = [
        'input[type="text"]',
        'input:not([type])',
        'input'
    ]

    for frame in page.frames:
        for selector in selectors:
            try:
                inputs = frame.locator(selector)
                count = await inputs.count()

                if count > 0:
                    # первое поле — номер карты
                    card_input = inputs.nth(0)
                    await card_input.scroll_into_view_if_needed(timeout=3000)
                    await card_input.fill(card_number, timeout=5000)

                    print(f"Номер карты введён: {card_number}")
                    return True
            except Exception:
                pass

    return False


async def save_captcha_image(page):
    """
    Ищем именно captcha, а не логотип.
    Сохраняем её как debug/site_captcha.png.
    """

    captcha_path = DEBUG_DIR / "site_captcha.png"
    debug_info_path = DEBUG_DIR / "captcha_candidates.txt"

    all_candidates = []

    for frame_index, frame in enumerate(page.frames):
        try:
            images = frame.locator("img")
            count = await images.count()

            print(f"Frame {frame_index}: найдено img: {count}")

            for i in range(count):
                img = images.nth(i)

                try:
                    box = await img.bounding_box()
                    src = await img.get_attribute("src")
                    alt = await img.get_attribute("alt")

                    if box is None:
                        continue

                    width = box["width"]
                    height = box["height"]
                    x = box["x"]
                    y = box["y"]

                    item = {
                        "frame_index": frame_index,
                        "index": i,
                        "img": img,
                        "width": width,
                        "height": height,
                        "x": x,
                        "y": y,
                        "src": src,
                        "alt": alt
                    }

                    all_candidates.append(item)

                except Exception:
                    pass

        except Exception:
            pass

    # Сохраняем все найденные картинки для отладки
    with open(debug_info_path, "w", encoding="utf-8") as f:
        for item in all_candidates:
            f.write(f"frame: {item['frame_index']}\n")
            f.write(f"index: {item['index']}\n")
            f.write(f"size: {item['width']}x{item['height']}\n")
            f.write(f"position: x={item['x']}, y={item['y']}\n")
            f.write(f"src: {item['src']}\n")
            f.write(f"alt: {item['alt']}\n")
            f.write("-" * 50 + "\n")

    # Фильтр именно под captcha:
    # логотип был 192x29, поэтому height >= 35
    captcha_candidates = []

    for item in all_candidates:
        width = item["width"]
        height = item["height"]
        src = item["src"] or ""
        alt = item["alt"] or ""

        # Отсекаем логотипы
        bad_words = ["logo", "logotype", "brand", "vk", "telegram", "whatsapp"]

        if any(word in src.lower() for word in bad_words):
            continue

        if any(word in alt.lower() for word in bad_words):
            continue

        # Captcha обычно имеет такую форму
        if 80 <= width <= 300 and 35 <= height <= 120:
            captcha_candidates.append(item)

    if not captcha_candidates:
        fallback_path = DEBUG_DIR / "captcha_fallback_form.png"
        await page.screenshot(path=str(fallback_path), full_page=True)

        raise Exception(
            "Не удалось найти captcha как отдельное изображение. "
            f"Скрин страницы сохранён: {fallback_path}. "
            f"Список картинок сохранён: {debug_info_path}"
        )

    # Берём кандидата с самой большой высотой,
    # потому что captcha обычно выше логотипов/иконок
    best = max(captcha_candidates, key=lambda item: item["height"])

    await best["img"].scroll_into_view_if_needed(timeout=3000)
    await best["img"].screenshot(path=str(captcha_path))

    print(f"Captcha сохранена: {captcha_path}")
    print(f"Размер captcha: {best['width']}x{best['height']}")
    print(f"src: {best['src']}")

    return str(captcha_path)


async def open_balance_page_accept_fill_and_get_captcha(card_number: str):
    """
    Открывает сайт, принимает условия, вводит номер карты
    и сохраняет captcha в debug/site_captcha.png.
    """

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=False,
            slow_mo=300
        )

        page = await browser.new_page(
            viewport={"width": 1280, "height": 900}
        )

        await page.goto(BALANCE_URL, wait_until="domcontentloaded")
        await page.wait_for_timeout(3000)

        await page.screenshot(
            path=str(DEBUG_DIR / "before_accept.png"),
            full_page=True
        )

        await click_cookie_accept(page)

        clicked = await click_accept_terms(page)

        if not clicked:
            await page.screenshot(
                path=str(DEBUG_DIR / "accept_failed.png"),
                full_page=True
            )
            await browser.close()
            raise Exception("Не удалось найти или нажать кнопку согласия")

        await page.wait_for_timeout(2000)

        await page.screenshot(
            path=str(DEBUG_DIR / "after_accept.png"),
            full_page=True
        )

        filled = await fill_card_number(page, card_number)

        if not filled:
            await page.screenshot(
                path=str(DEBUG_DIR / "fill_card_failed.png"),
                full_page=True
            )
            await browser.close()
            raise Exception("Не удалось найти поле для ввода номера карты")

        await page.wait_for_timeout(1000)

        await page.screenshot(
            path=str(DEBUG_DIR / "after_fill_card.png"),
            full_page=True
        )

        captcha_path = await save_captcha_image(page)

        await page.wait_for_timeout(2000)
        await browser.close()

        return captcha_path


async def open_balance_page_accept_and_get_captcha():
    """
    Открывает сайт, принимает условия и сохраняет captcha.
    Номер карты НЕ вводится.
    Используется только для сбора датасета captcha.
    """

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=False,
            slow_mo=300
        )

        page = await browser.new_page(
            viewport={"width": 1280, "height": 900}
        )

        await page.goto(BALANCE_URL, wait_until="domcontentloaded")
        await page.wait_for_timeout(3000)

        await page.screenshot(
            path=str(DEBUG_DIR / "before_accept.png"),
            full_page=True
        )

        await click_cookie_accept(page)

        clicked = await click_accept_terms(page)

        if not clicked:
            await page.screenshot(
                path=str(DEBUG_DIR / "accept_failed.png"),
                full_page=True
            )
            await browser.close()
            raise Exception("Не удалось найти или нажать кнопку согласия")

        await page.wait_for_timeout(2000)

        await page.screenshot(
            path=str(DEBUG_DIR / "after_accept.png"),
            full_page=True
        )

        captcha_path = await save_captcha_image(page)

        await page.wait_for_timeout(1000)
        await browser.close()

        return captcha_path


class BalanceCheckSession:
    def __init__(self, playwright, browser, page):
        self.playwright = playwright
        self.browser = browser
        self.page = page

    async def close(self):
        try:
            await self.browser.close()
        except Exception:
            pass

        try:
            await self.playwright.stop()
        except Exception:
            pass


async def create_balance_session_and_get_captcha(card_number: str):
    """
    Открывает сайт, принимает условия, вводит номер карты,
    получает captcha и НЕ закрывает браузер.
    Возвращает session и путь к captcha.
    """

    playwright = await async_playwright().start()

    browser = await playwright.chromium.launch(
        headless=False,
        slow_mo=300
    )

    page = await browser.new_page(
        viewport={"width": 1280, "height": 900}
    )

    await page.goto(BALANCE_URL, wait_until="domcontentloaded")
    await page.wait_for_timeout(3000)

    await page.screenshot(
        path=str(DEBUG_DIR / "before_accept.png"),
        full_page=True
    )

    await click_cookie_accept(page)

    clicked = await click_accept_terms(page)

    if not clicked:
        await page.screenshot(
            path=str(DEBUG_DIR / "accept_failed.png"),
            full_page=True
        )
        await browser.close()
        await playwright.stop()
        raise Exception("Не удалось найти или нажать кнопку согласия")

    await page.wait_for_timeout(2000)

    await page.screenshot(
        path=str(DEBUG_DIR / "after_accept.png"),
        full_page=True
    )

    filled = await fill_card_number(page, card_number)

    if not filled:
        await page.screenshot(
            path=str(DEBUG_DIR / "fill_card_failed.png"),
            full_page=True
        )
        await browser.close()
        await playwright.stop()
        raise Exception("Не удалось найти поле для ввода номера карты")

    await page.wait_for_timeout(1000)

    await page.screenshot(
        path=str(DEBUG_DIR / "after_fill_card.png"),
        full_page=True
    )

    captcha_path = await save_captcha_image(page)

    session = BalanceCheckSession(
        playwright=playwright,
        browser=browser,
        page=page
    )

    return session, captcha_path


async def fill_captcha_code(page, captcha_code: str):
    """
    Вводит код captcha во второе текстовое поле формы.
    Первое поле — номер карты.
    Второе поле — код проверки.
    """

    selectors = [
        'input[type="text"]',
        'input:not([type])',
        'input'
    ]

    for frame in page.frames:
        for selector in selectors:
            try:
                inputs = frame.locator(selector)
                count = await inputs.count()

                if count >= 2:
                    captcha_input = inputs.nth(1)
                    await captcha_input.scroll_into_view_if_needed(timeout=3000)
                    await captcha_input.fill(captcha_code, timeout=5000)

                    print(f"Код captcha введён: {captcha_code}")
                    return True

            except Exception:
                pass

    return False


async def click_execute_request(page):
    """
    Нажимает кнопку 'Выполнить запрос'.
    """

    selectors = [
        'input[type="submit"][value*="Выполнить запрос"]',
        'input[type="button"][value*="Выполнить запрос"]',
        'button:has-text("Выполнить запрос")',
        'text=Выполнить запрос'
    ]

    for frame in page.frames:
        for selector in selectors:
            try:
                locator = frame.locator(selector).first

                if await locator.count() > 0:
                    await locator.scroll_into_view_if_needed(timeout=3000)
                    await locator.click(timeout=5000)

                    print("Кнопка 'Выполнить запрос' нажата")
                    return True

            except Exception:
                pass

    return False


async def submit_captcha_and_get_result(session: BalanceCheckSession, captcha_code: str):
    """
    Вводит captcha, нажимает 'Выполнить запрос',
    сохраняет скрин результата и возвращает текст страницы.
    """

    page = session.page

    filled = await fill_captcha_code(page, captcha_code)

    if not filled:
        await page.screenshot(
            path=str(DEBUG_DIR / "captcha_fill_failed.png"),
            full_page=True
        )
        raise Exception("Не удалось найти поле для ввода captcha")

    await page.wait_for_timeout(1000)

    await page.screenshot(
        path=str(DEBUG_DIR / "after_fill_captcha.png"),
        full_page=True
    )

    clicked = await click_execute_request(page)

    if not clicked:
        await page.screenshot(
            path=str(DEBUG_DIR / "execute_failed.png"),
            full_page=True
        )
        raise Exception("Не удалось нажать кнопку 'Выполнить запрос'")

    await page.wait_for_timeout(4000)

    await page.screenshot(
        path=str(DEBUG_DIR / "after_submit_result.png"),
        full_page=True
    )

    result_text_parts = []

    for frame in page.frames:
        try:
            text = await frame.locator("body").inner_text(timeout=3000)
            if text.strip():
                result_text_parts.append(text.strip())
        except Exception:
            pass

    result_text = "\n\n".join(result_text_parts)

    result_path = DEBUG_DIR / "result_text.txt"

    with open(result_path, "w", encoding="utf-8") as f:
        f.write(result_text)

    print("Результат сохранён:", result_path)

    return result_text