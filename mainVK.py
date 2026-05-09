import logging
import os
import vk_api
from vk_api.longpoll import VkLongPoll, VkEventType
from dotenv import load_dotenv

from orchestrator import DialogOrchestrator

load_dotenv()
logging.basicConfig(level=logging.INFO)

vk_session = vk_api.VkApi(token=os.getenv("VK_GROUP_TOKEN"))
vk = vk_session.get_api()
longpoll = VkLongPoll(vk_session)
orchestrator = DialogOrchestrator()


def send_message(user_id: int, text: str) -> None:
    """Отправляет сообщение пользователю ВКонтакте."""
    vk.messages.send(
        user_id=user_id,
        message=text,
        random_id=0,
    )


def handle_message(event) -> None:
    """Обрабатывает входящее текстовое сообщение."""
    user_id = event.user_id
    user_text = event.text

    if user_text.strip().lower() in ("/start", "начать", "start"):
        send_message(user_id, "Здравствуйте! Я — ИИ-ассистент службы поддержки. Чем могу помочь?")
        return

    result = orchestrator.process_message(user_id, user_text)

    if result["action"] == "escalate":
        send_message(user_id, "Сейчас я передам вас живому оператору. Пожалуйста, ожидайте.")
        logging.info(f"Эскалация для {user_id}: {result['reason']}")
    else:
        send_message(user_id, result["response"])


def main() -> None:
    logging.info("Бот запущен и ожидает сообщений...")
    for event in longpoll.listen():
        if event.type == VkEventType.MESSAGE_NEW and event.to_me:
            try:
                handle_message(event)
            except Exception as e:
                logging.error(f"Ошибка обработки сообщения от {event.user_id}: {e}")


if __name__ == "__main__":
    main()