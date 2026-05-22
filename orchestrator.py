# orchestrator.py
import json
import asyncio
from gigachat_client import SupportAssistant
from vision_analyzer import VisionAnalyzer

class DialogOrchestrator:
    def __init__(self):
        self.assistant = SupportAssistant()
        self.vision = VisionAnalyzer()
        self.conversation_history = {}

        try:
            with open("system_prompt.md", encoding="utf-8") as f:
                self.system_prompt = f.read()
        except FileNotFoundError:
            from prompts import SYSTEM_PROMPT
            self.system_prompt = SYSTEM_PROMPT

    # Для текста
    def process_message(self, user_id: int, user_message: str) -> dict:
        return self._process(user_id, user_message)

    async def process_message_async(self, user_id: int, user_message: str) -> dict:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._process, user_id, user_message)

    def _process(self, user_id: int, user_message: str) -> dict:
        full_prompt = (self.system_prompt + "\n\n" + self._format_history(user_id) +
                       f"\nПользователь: {user_message}")
        raw_response = self.assistant.get_response(full_prompt)
        try:
            decision = json.loads(raw_response)
            if decision.get("escalate"):
                return {
                    "action": "escalate",
                    "reason": decision.get("reason", "Не указана"),
                    "operator_message": f"Пользователь {user_id} просит помощи. Причина: {decision.get('reason')}",
                }
        except json.JSONDecodeError:
            pass
        self._save_to_history(user_id, user_message, raw_response)
        return {"action": "message", "response": raw_response}

    # Для медиа 
    def process_photo(self, user_id: int, photo_bytes: bytes, caption: str = "") -> dict:
        """Анализирует фото и возвращает ответ."""
        # 1. Анализ изображения
        analysis = self.vision.analyze_image(photo_bytes)
        # 2. Формируем текстовое описание для LLM
        if analysis.get("damage_detected"):
            desc = f"Пользователь прислал фото. Результат анализа: {analysis.get('description')}"
        else:
            desc = f"Пользователь прислал фото. Модуль CV не смог определить повреждение. {analysis.get('message', '')}"
        # 3. Объединяем с caption (текстом от пользователя)
        full_text = f"{desc}\nПодпись к фото: {caption}" if caption else desc
        # 4. Передаём в обычный текстовый обработчик
        return self._process(user_id, full_text)

    async def process_photo_async(self, user_id: int, photo_bytes: bytes, caption: str = "") -> dict:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.process_photo, user_id, photo_bytes, caption)

    def process_video(self, user_id: int, caption: str = "") -> dict:
        """Видео пока не анализируем – просто эскалируем."""
        return {
            "action": "escalate",
            "reason": "Видеофайл – требуется просмотр оператором",
            "operator_message": f"Пользователь {user_id} прислал видео. Требуется ручная обработка."
        }

    async def process_video_async(self, user_id: int, caption: str = "") -> dict:
        return self.process_video(user_id, caption)

    # --- вспомогательные методы (без изменений) ---
    def _format_history(self, user_id: int) -> str:
        history = self.conversation_history.get(user_id, [])
        lines = []
        for entry in history[-6:]:
            lines.append(f"Пользователь: {entry['user']}\nАссистент: {entry['assistant']}")
        return "\n".join(lines)

    def _save_to_history(self, user_id: int, user_msg: str, assistant_msg: str) -> None:
        if user_id not in self.conversation_history:
            self.conversation_history[user_id] = []
        self.conversation_history[user_id].append({"user": user_msg, "assistant": assistant_msg})
        self.conversation_history[user_id] = self.conversation_history[user_id][-10:]