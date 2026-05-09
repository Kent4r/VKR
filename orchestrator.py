# orchestrator.py
import json
import asyncio
from gigachat_client import SupportAssistant


class DialogOrchestrator:
    def __init__(self):
        self.assistant = SupportAssistant()
        self.conversation_history = {}  # {user_id: [{"user": ..., "assistant": ...}]}

        # Загружаем системный промпт из файла — так сложнее провести prompt injection
        try:
            with open("system_prompt.md", encoding="utf-8") as f:
                self.system_prompt = f.read()
        except FileNotFoundError:
            # Фолбэк на prompts.py если md-файла ваще нет
            from prompts import SYSTEM_PROMPT
            self.system_prompt = SYSTEM_PROMPT

    # ── публичные методы ──────────────────────────────────────────────────────

    def process_message(self, user_id: int, user_message: str) -> dict:
        """Синхронный метод — основной.
        Вызывай напрямую в VK-боте (Long Poll).
        В aiogram используй process_message_async."""
        return self._process(user_id, user_message)

    async def process_message_async(self, user_id: int, user_message: str) -> dict:
        """Async-обёртка для aiogram: запускает синхронный _process
        в пуле потоков, не блокируя event loop."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._process, user_id, user_message)

    # ── внутренняя логика ─────────────────────────────────────────────────────

    def _process(self, user_id: int, user_message: str) -> dict:
        """Общая логика обработки: формирует промпт, вызывает LLM, парсит ответ."""
        full_prompt = (
            self.system_prompt
            + "\n\n"
            + self._format_history(user_id)
            + f"\nПользователь: {user_message}"
        )

        raw_response = self.assistant.get_response(full_prompt)

        try:
            decision = json.loads(raw_response)
            if decision.get("escalate"):
                return {
                    "action": "escalate",
                    "reason": decision.get("reason", "Причина не указана"),
                    "operator_message": (
                        f"Пользователь {user_id} просит помощи. "
                        f"Причина: {decision.get('reason')}"
                    ),
                }
        except json.JSONDecodeError:
            pass

        self._save_to_history(user_id, user_message, raw_response)
        return {"action": "message", "response": raw_response}

    def _format_history(self, user_id: int) -> str:
        history = self.conversation_history.get(user_id, [])
        lines = []
        for entry in history[-6:]:  # последние 5 обменов = 6 записей
            lines.append(f"Пользователь: {entry['user']}\nАссистент: {entry['assistant']}")
        return "\n".join(lines)

    def _save_to_history(self, user_id: int, user_msg: str, assistant_msg: str) -> None:
        if user_id not in self.conversation_history:
            self.conversation_history[user_id] = []
        self.conversation_history[user_id].append({
            "user": user_msg,
            "assistant": assistant_msg,
        })
        # Держим только последние 10 обменов в памяти
        self.conversation_history[user_id] = self.conversation_history[user_id][-10:]