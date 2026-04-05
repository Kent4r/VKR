# orchestrator.py
import json
from gigachat_client import SupportAssistant
from prompts import SYSTEM_PROMPT

class DialogOrchestrator:
    def __init__(self):
        self.assistant = SupportAssistant()
        self.conversation_history = {}  # Пока храним в памяти, позже можно заменить на БД

    async def process_message(self, user_id: int, user_message: str):
        """Обрабатывает входящее сообщение и решает, что с ним делать."""
        
        # 1. Добавляем системный промпт и историю в запрос к LLM
        full_prompt = SYSTEM_PROMPT + "\n\n" + self._format_history(user_id) + f"\nПользователь: {user_message}"
        
        # 2. Получаем ответ от модели (GigaChat с RAG)
        raw_response = self.assistant.get_response(full_prompt)
        
        # 3. Проверяем, не хочет ли модель передать диалог оператору
        try:
            # Пытаемся распарсить ответ как JSON
            decision = json.loads(raw_response)
            if decision.get("escalate"):
                # Эскалация!
                return {
                    "action": "escalate",
                    "reason": decision.get("reason", "Причина не указана"),
                    "operator_message": f"Пользователь {user_id} просит помощи. Причина: {decision.get('reason')}"
                }
        except json.JSONDecodeError:
            # Ответ не JSON — это обычное сообщение от ассистента
            pass
        
        # 4. Сохраняем сообщение и ответ в историю
        self._save_to_history(user_id, user_message, raw_response)
        
        # 5. Возвращаем обычный ответ
        return {
            "action": "message",
            "response": raw_response
        }
    
    def _format_history(self, user_id):
        history = self.conversation_history.get(user_id, [])
        formatted = []
        for entry in history[-6:]:
            formatted.append(f"Пользователь: {entry['user']}\nАссистент: {entry['assistant']}")
        return "\n".join(formatted)

    def _save_to_history(self, user_id, user_msg, assistant_msg):
        if user_id not in self.conversation_history:
            self.conversation_history[user_id] = []
        self.conversation_history[user_id].append({
            "user": user_msg,
            "assistant": assistant_msg
        })