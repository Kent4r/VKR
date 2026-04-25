import asyncio
import logging
import os
from dotenv import load_dotenv
from telethon import TelegramClient, events
from telethon.network.connection import ConnectionTcpMTProxyRandomizedIntermediate
from orchestrator import DialogOrchestrator

load_dotenv()
logging.basicConfig(level=logging.INFO)

# Конфигурация
API_ID = int(os.getenv("API_ID"))
API_HASH = os.getenv("API_HASH")
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# Данные MTProto прокси
PROXY_HOST = "127.0.0.1"
PROXY_PORT = 1444
PROXY_SECRET = os.getenv("PROXY_SECRET")

# Создаём клиента Telethon
client = TelegramClient(
    "bot_session",                     # имя файла сессии
    API_ID,
    API_HASH,
    connection=ConnectionTcpMTProxyRandomizedIntermediate,
    proxy=(PROXY_HOST, PROXY_PORT, PROXY_SECRET)
)

orchestrator = DialogOrchestrator()

# --- Обработчики событий ---
@client.on(events.NewMessage(pattern="/start"))
async def start_handler(event):
    await event.reply("Здравствуйте! Я — ИИ-ассистент службы поддержки. Чем могу помочь?")

@client.on(events.NewMessage)
async def message_handler(event):
    # Игнорируем сообщения от самого бота
    if event.out:
        return
    user_id = event.sender_id
    user_text = event.raw_text

    result = await orchestrator.process_message(user_id, user_text)

    if result["action"] == "escalate":
        await event.reply("⚠️ Сейчас я передам вас живому оператору. Пожалуйста, ожидайте.")
        logging.info(f"Эскалация для {user_id}: {result['reason']}")
    else:
        await event.reply(result["response"], parse_mode="markdown")

# --- Запуск ---
async def main():
    await client.start(bot_token=BOT_TOKEN)
    logging.info("Бот запущен и работает через MTProto прокси")
    await client.run_until_disconnected()

if __name__ == "__main__":
    asyncio.run(main())