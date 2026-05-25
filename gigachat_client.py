# gigachat_client.py
import os
import logging
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_gigachat import GigaChat

load_dotenv()
logger = logging.getLogger(__name__)

class SupportAssistant:
    def __init__(self, docs_path: str = "./database/docs"):
        self.llm = GigaChat(
            credentials=os.getenv("GIGACHAT_API_KEY"),
            model="GigaChat",
            verify_ssl_certs=False,
            temperature=0.7,   # чуть выше для разнообразия
            timeout=30
        )
        self.vectorstore = None
        if os.path.exists(docs_path) and any(os.scandir(docs_path)):
            logger.info("Загрузка документов...")
            documents = []
            for file in os.listdir(docs_path):
                file_path = os.path.join(docs_path, file)
                if file.endswith(".txt"):
                    loader = TextLoader(file_path, encoding='utf-8')
                    documents.extend(loader.load())
                elif file.endswith(".pdf"):
                    loader = PyPDFLoader(file_path)
                    documents.extend(loader.load())
            if documents:
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                docs = text_splitter.split_documents(documents)
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
                self.vectorstore = Chroma.from_documents(docs, embeddings)
                logger.info("Векторная БД создана")
            else:
                logger.warning("Нет документов для индексации")
        else:
            logger.warning("Папка docs пуста")

    def get_response(self, full_prompt: str) -> str:
        """
        full_prompt — это уже полностью сформированный промпт (системный + история + вопрос).
        Мы только дополняем его контекстом из документации.
        """
        if not self.vectorstore:
            # Нет RAG — вызываем как есть
            response = self.llm.invoke(full_prompt)
            return response.content if hasattr(response, 'content') else str(response)

        # Есть RAG — извлекаем последний вопрос пользователя из промпта (это грубо, но работает)
        # Проще: ищем строку "Пользователь:" в конце промпта
        lines = full_prompt.split('\n')
        user_question = ""
        for i in range(len(lines)-1, -1, -1):
            if lines[i].startswith("Пользователь:"):
                user_question = lines[i].replace("Пользователь:", "").strip()
                break
        if not user_question:
            user_question = full_prompt  # fallback

        # Поиск релевантных документов
        docs = self.vectorstore.similarity_search(user_question, k=4)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Вставляем контекст в промпт после системной части
        # Ищем место для вставки (например, перед "История диалога" или перед последним вопросом)
        # Упростим: добавим в начало
        enhanced_prompt = f"{full_prompt}\n\nДОПОЛНИТЕЛЬНЫЙ КОНТЕКСТ ИЗ ДОКУМЕНТАЦИИ:\n{context}\n\nОтветь, используя этот контекст."
        
        response = self.llm.invoke(enhanced_prompt)
        return response.content if hasattr(response, 'content') else str(response)