# gigachat_client.py
import os
from dotenv import load_dotenv
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_gigachat import GigaChat

load_dotenv()

class SupportAssistant:
    def __init__(self, docs_path: str = "./database/docs"):
        # 1. Инициализация модели GigaChat
        self.llm = GigaChat(
            credentials=os.getenv("GIGACHAT_API_KEY"),
            model="GigaChat",
            verify_ssl_certs=False,
            temperature=0.3,  # Низкая температура для фактологичности
            timeout=30
        )

        # 2. Создание и загрузка векторной базы знаний (RAG)
        self.vectorstore = None
        if os.path.exists(docs_path) and any(os.scandir(docs_path)):
            print("🔄 Загрузка и индексация документов...")
            documents = []
            for file in os.listdir(docs_path):
                if file.endswith(".txt"):
                    loader = TextLoader(os.path.join(docs_path, file), encoding='utf-8')
                elif file.endswith(".pdf"):
                    loader = PyPDFLoader(os.path.join(docs_path, file))
                else:
                    continue
                documents.extend(loader.load())
            
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            docs = text_splitter.split_documents(documents)
            # Используем бесплатную эмбеддинг-модель
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
            self.vectorstore = Chroma.from_documents(docs, embeddings)
            print("✅ База знаний готова!")

        # 3. Настройка памяти для диалога
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=5  # Помнить последние 5 обменов сообщениями
        )

    def get_response(self, user_message: str) -> str:
        """Получает ответ от GigaChat, дополняя его контекстом из документации."""
        if not self.vectorstore:
            # Если БД нет, просто отправляем промпт в GigaChat
            return self.llm.invoke(user_message)
        
        # Создаём цепочку RAG
        qa_chain = ConversationalRetrievalChain.from_llm(
            self.llm,
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 4}),
            memory=self.memory
        )
        
        result = qa_chain({"question": user_message})
        return result["answer"]