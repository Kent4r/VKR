# mainVK.py
import logging
import os
import re
import requests
from urllib.parse import urlparse
import vk_api
from vk_api.longpoll import VkLongPoll, VkEventType
from dotenv import load_dotenv

from orchestrator import DialogOrchestrator

load_dotenv()
logging.basicConfig(level=logging.INFO)

VK_GROUP_TOKEN = os.getenv("VK_GROUP_TOKEN")
vk_session = vk_api.VkApi(token=VK_GROUP_TOKEN)
vk = vk_session.get_api()
longpoll = VkLongPoll(vk_session)
orchestrator = DialogOrchestrator()


def send_message(user_id: int, text: str) -> None:
    vk.messages.send(user_id=user_id, message=text, random_id=0)


def download_photo_by_url(url: str) -> bytes:
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.content


def get_photo_url_from_string(attach_str: str) -> str:
    """
    Парсит строку типа 'photo123_456' и возвращает URL самой большой фотографии.
    """
    match = re.match(r'(photo)(-?\d+)_(\d+)', attach_str)
    if not match:
        return None
    prefix, owner_id, media_id = match.groups()
    # owner_id может быть отрицательным (группа)
    photo_id = f"{owner_id}_{media_id}"
    try:
        # Получаем информацию о фото
        photos_info = vk.photos.getById(photos=photo_id)
        if not photos_info:
            return None
        photo_data = photos_info[0]
        sizes = photo_data.get('sizes', [])
        if not sizes:
            # Если нет sizes, берём прямой url (старый формат)
            return photo_data.get('url')
        # Сортируем по убыванию ширины
        sizes_sorted = sorted(sizes, key=lambda x: x.get('width', 0), reverse=True)
        return sizes_sorted[0]['url']
    except Exception as e:
        logging.error(f"Ошибка получения URL фото {photo_id}: {e}")
        return None


def handle_attachments(user_id: int, attachments: list, caption: str) -> dict:
    """Обрабатывает первое фото или видео из вложений (список может содержать строки или словари)."""
    for attach in attachments:
        # Если attach — словарь (новый формат)
        if isinstance(attach, dict):
            attach_type = attach.get('type')
            if attach_type == 'photo':
                photo_url = None
                # Пытаемся получить URL из sizes
                photo_obj = attach.get('photo', {})
                sizes = photo_obj.get('sizes', [])
                if sizes:
                    sizes_sorted = sorted(sizes, key=lambda x: x.get('width', 0), reverse=True)
                    photo_url = sizes_sorted[0]['url']
                if not photo_url:
                    # fallback: прямой url
                    photo_url = photo_obj.get('url')
                if photo_url:
                    try:
                        photo_bytes = download_photo_by_url(photo_url)
                        return orchestrator.process_photo(user_id, photo_bytes, caption)
                    except Exception as e:
                        logging.error(f"Ошибка скачивания фото: {e}")
                        return {
                            "action": "escalate",
                            "reason": f"Не удалось загрузить фото: {e}",
                            "operator_message": f"Ошибка загрузки фото от {user_id}"
                        }
                else:
                    return {
                        "action": "escalate",
                        "reason": "Не удалось получить ссылку на фото",
                        "operator_message": f"Нет ссылки на фото от {user_id}"
                    }
            elif attach_type == 'video':
                return orchestrator.process_video(user_id, caption)
            # Другие типы вложений игнорируем

        # Если attach — строка (старый формат)
        elif isinstance(attach, str):
            if attach.startswith('photo'):
                photo_url = get_photo_url_from_string(attach)
                if photo_url:
                    try:
                        photo_bytes = download_photo_by_url(photo_url)
                        return orchestrator.process_photo(user_id, photo_bytes, caption)
                    except Exception as e:
                        logging.error(f"Ошибка скачивания фото: {e}")
                        return {
                            "action": "escalate",
                            "reason": f"Не удалось загрузить фото: {e}",
                            "operator_message": f"Ошибка загрузки фото от {user_id}"
                        }
                else:
                    return {
                        "action": "escalate",
                        "reason": "Не удалось получить URL фото из строки",
                        "operator_message": f"Проблема с фото-строкой {attach}"
                    }
            elif attach.startswith('video'):
                return orchestrator.process_video(user_id, caption)
            # Игнорируем другие типы строк (doc, audio и т.п.)

    return None  # Если не нашли фото/видео


def handle_message(event) -> None:
    user_id = event.user_id
    text = event.text or ""
    attachments = event.attachments or []

    # Команда старт
    if text.strip().lower() in ("/start", "начать", "start"):
        send_message(user_id, "Здравствуйте! Я — ИИ-ассистент службы поддержки. Чем могу помочь?")
        return

    # Если есть вложения — обрабатываем первое фото/видео
    if attachments:
        result = handle_attachments(user_id, attachments, text)
        if result:
            if result["action"] == "escalate":
                send_message(user_id, "⚠️ Сейчас я передам вас живому оператору. Пожалуйста, ожидайте.")
                logging.info(f"Эскалация для {user_id} (медиа): {result['reason']}")
            else:
                send_message(user_id, result["response"])
            return

    # Обычный текст
    result = orchestrator.process_message(user_id, text)
    if result["action"] == "escalate":
        send_message(user_id, "⚠️ Сейчас я передам вас живому оператору. Пожалуйста, ожидайте.")
        logging.info(f"Эскалация для {user_id}: {result['reason']}")
    else:
        send_message(user_id, result["response"])


def main() -> None:
    logging.info("VK-бот запущен и ожидает сообщений...")
    for event in longpoll.listen():
        if event.type == VkEventType.MESSAGE_NEW and event.to_me:
            try:
                handle_message(event)
            except Exception as e:
                logging.error(f"Ошибка обработки сообщения от {event.user_id}: {e}")


if __name__ == "__main__":
    main()