# orchestrator.py
import json
import asyncio
from gigachat_client import SupportAssistant
from vision_analyzer import VisionAnalyzer
from datetime import datetime

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
        # Получаем историю для этого пользователя (последние 5 обменов)
        history = self.conversation_history.get(user_id, [])
        chat_history = [(entry["user"], entry["assistant"]) for entry in history[-5:]]

        # Вызов LLM с системным промптом и историей
        raw_response = self.assistant.get_response(
            user_message=user_message,
            system_prompt=self.system_prompt,
            chat_history=chat_history
        )

        # Логирование
        with open("dialog_log.txt", "a", encoding="utf-8") as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"User {user_id} at {datetime.now()}\n")
            f.write(f"Q: {user_message}\n")
            f.write(f"A: {raw_response[:500]}\n")
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
        analysis = self.vision.analyze_image(photo_bytes)
        cv_text = analysis.get("description", "")
        objects_list = analysis.get("objects", [])
        if objects_list:
            obj_descs = [f"{obj.get('object')} (уверенность {obj.get('confidence', 0):.2f})" for obj in objects_list]
            cv_text += f"\nДетектированные объекты: {', '.join(obj_descs)}."
        if analysis.get("recommendation"):
            cv_text += f"\nРекомендация CV: {analysis['recommendation']}"
        full_user_message = f"{cv_text}\n\nСообщение пользователя: {caption}" if caption else cv_text
        return self._process(user_id, full_user_message)

    async def process_photo_async(self, user_id: int, photo_bytes: bytes, caption: str = "") -> dict:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.process_photo, user_id, photo_bytes, caption)

    # --- Видео (сразу эскалация) ---
    def process_video(self, user_id: int, caption: str = "") -> dict:
        return {
            "action": "escalate",
            "reason": "Видеофайл – требуется просмотр оператором",
            "operator_message": f"Пользователь {user_id} прислал видео. Требуется ручная обработка."
        }

    async def process_video_async(self, user_id: int, caption: str = "") -> dict:
        return self.process_video(user_id, caption)

    # --- Вспомогательные методы ---
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