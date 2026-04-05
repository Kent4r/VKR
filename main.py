# main.py
import asyncio
import logging
from aiogram import Bot, Dispatcher, types
from aiogram.filters import CommandStart
from dotenv import load_dotenv
import os

from orchestrator import DialogOrchestrator

load_dotenv()
logging.basicConfig(level=logging.INFO)

bot = Bot(token=os.getenv("TELEGRAM_BOT_TOKEN"))
dp = Dispatcher()
orchestrator = DialogOrchestrator()

@dp.message(CommandStart())
async def start(message: types.Message):
    await message.answer("Здравствуйте! Я — ИИ-ассистент службы поддержки. Чем могу помочь?")

@dp.message()
async def handle_message(message: types.Message):
    user_id = message.from_user.id
    user_text = message.text
    
    # Отправляем сообщение в оркестратор
    result = await orchestrator.process_message(user_id, user_text)
    
    if result["action"] == "escalate":
        # Здесь можно отправить уведомление оператору в админ-панель
        await message.answer("⚠️ Сейчас я передам вас живому оператору. Пожалуйста, ожидайте.")
        logging.info(f"Эскалация для {user_id}: {result['reason']}")
    else:
        await message.answer(result["response"],parse_mode='Markdown')

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())