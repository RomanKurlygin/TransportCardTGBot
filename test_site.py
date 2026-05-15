import asyncio
from balance_site import open_balance_page_accept_fill_and_get_captcha


async def main():
    card_number = "3715354948"

    captcha_path = await open_balance_page_accept_fill_and_get_captcha(card_number)

    print("Captcha сохранена:")
    print(captcha_path)


if __name__ == "__main__":
    asyncio.run(main())