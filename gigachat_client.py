# gigachat_client.py
import os
import logging
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_gigachat import GigaChat
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()
logger = logging.getLogger(__name__)

CATEGORY_LABELS = {
    "general":         "Общие инструкции",
    "routers":         "Документация по роутерам",
    "terminals":       "Документация по ONU-терминалам",
    "troubleshooting": "Диагностика и устранение неисправностей",
}

class SupportAssistant:
    def __init__(self, docs_path: str = "./database/docs"):
        # Инициализация LLM
        self.llm = GigaChat(
            credentials=os.getenv("GIGACHAT_API_KEY"),
            model="GigaChat-2",
            verify_ssl_certs=False,
            temperature=0.3,
            timeout=30
        )
        logger.info("GigaChat инициализирован")

        # Загрузка и индексация документов (RAG)
        self.vectorstore = None
        documents = []
        if os.path.exists(docs_path):
            for root, dirs, files in os.walk(docs_path):
                folder_name = os.path.basename(root)
                category = CATEGORY_LABELS.get(folder_name, folder_name)
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        if file.endswith(".txt"):
                            loader = TextLoader(file_path, encoding="utf-8")
                        elif file.endswith(".pdf"):
                            loader = PyPDFLoader(file_path)
                        else:
                            continue
                        loaded = loader.load()
                        for doc in loaded:
                            doc.metadata["category"] = category
                            doc.metadata["source_file"] = file
                        documents.extend(loaded)
                        logger.info("Загружен [%s] %s", category, file)
                    except Exception as e:
                        logger.error("Ошибка загрузки %s: %s", file, e)

        if documents:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", " ", ""]
            )
            docs = splitter.split_documents(documents)
            logger.info("Создано %d чанков", len(docs))
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )
            self.vectorstore = Chroma.from_documents(docs, embeddings)
            logger.info("Векторная БД создана")
        else:
            logger.warning("Документы не найдены. RAG отключён.")

    def get_response(self, user_message: str, system_prompt: str, chat_history: list = None) -> str:
        # 1. Находим релевантный контекст в документации
        context = ""
        if self.vectorstore:
            relevant_docs = self.vectorstore.similarity_search(user_message, k=4)
            if relevant_docs:
                context = "\n\n".join([doc.page_content for doc in relevant_docs])

        # 2. Формируем единое системное сообщение
        full_system_prompt = system_prompt
        if context:
            full_system_prompt += f"\n\nВот актуальная информация из документации, используй её для ответа:\n{context}"

        # 3. Собираем список сообщений для отправки в API
        messages = [SystemMessage(content=full_system_prompt)]

        # 4. Добавляем историю диалога (чередование Human/AI)
        if chat_history:
            for human_msg, ai_msg in chat_history:
                messages.append(HumanMessage(content=human_msg))
                messages.append(AIMessage(content=ai_msg))

        # 5. Добавляем текущий запрос пользователя
        messages.append(HumanMessage(content=user_message))

        # 6. Отправляем запрос в GigaChat
        response = self.llm.invoke(messages)
        return response.content