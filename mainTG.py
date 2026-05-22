# mainTG.py (фрагмент с изменениями)
import asyncio
import logging
import os
from dotenv import load_dotenv
from telethon import TelegramClient, events
from telethon.network.connection import ConnectionTcpMTProxyRandomizedIntermediate
from orchestrator import DialogOrchestrator

load_dotenv()
logging.basicConfig(level=logging.INFO)

API_ID = int(os.getenv("API_ID"))
API_HASH = os.getenv("API_HASH")
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

PROXY_HOST = os.getenv("PROXY_HOST")
PROXY_PORT = int(os.getenv("PROXY_PORT"))
PROXY_SECRET = os.getenv("PROXY_SECRET")

client = TelegramClient(
    "bot_session",
    API_ID,
    API_HASH,
    connection=ConnectionTcpMTProxyRandomizedIntermediate,
    proxy=(PROXY_HOST, PROXY_PORT, PROXY_SECRET)
)

orchestrator = DialogOrchestrator()

@client.on(events.NewMessage(pattern="/start"))
async def start_handler(event):
    await event.reply("Здравствуйте! Я — ИИ-ассистент службы поддержки. Чем могу помочь?")

@client.on(events.NewMessage)
async def message_handler(event):
    if event.out:
        return
    user_id = event.sender_id

    # Обработка фото
    if event.photo:
        # Скачиваем фото в байты
        photo_bytes = await event.download_media(bytes)
        caption = event.raw_text or ""
        result = await orchestrator.process_photo_async(user_id, photo_bytes, caption)
        if result["action"] == "escalate":
            await event.reply("⚠️ Сейчас я передам вас живому оператору. Пожалуйста, ожидайте.")
            logging.info(f"Эскалация для {user_id} (фото): {result['reason']}")
        else:
            await event.reply(result["response"], parse_mode="markdown")
        return

    # Обработка видео
    if event.video:
        # Видео не анализируем, сразу эскалация
        caption = event.raw_text or ""
        result = orchestrator.process_video(user_id, caption)  # синхронный вызов
        if result["action"] == "escalate":
            await event.reply("⚠️ Видеофайл получен. Для его анализа нужен оператор. Ожидайте.")
            logging.info(f"Эскалация для {user_id} (видео): {result['reason']}")
        else:
            await event.reply(result["response"])
        return

    # Обычный текст
    user_text = event.raw_text
    if not user_text:
        return  # игнорируем стикеры и т.п.
    result = await orchestrator.process_message_async(user_id, user_text)
    if result["action"] == "escalate":
        await event.reply("⚠️ Сейчас я передам вас живому оператору. Пожалуйста, ожидайте.")
        logging.info(f"Эскалация для {user_id}: {result['reason']}")
    else:
        await event.reply(result["response"], parse_mode="markdown")

async def main():
    await client.start(bot_token=BOT_TOKEN)
    logging.info("Бот запущен и работает через MTProto прокси")
    await client.run_until_disconnected()

if __name__ == "__main__":
    asyncio.run(main())