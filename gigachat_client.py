# gigachat_client.py
import os
import logging
from dotenv import load_dotenv
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_gigachat import GigaChat

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SupportAssistant:
    def __init__(self, docs_path: str = "./database/docs"):
        # 1. GigaChat
        self.llm = GigaChat(
            credentials=os.getenv("GIGACHAT_API_KEY"),
            model="GigaChat",
            verify_ssl_certs=False,
            temperature=0.3,
            timeout=30
        )

        # 2. Векторная БД (RAG)
        self.vectorstore = None
        if os.path.exists(docs_path) and any(os.scandir(docs_path)):
            logger.info("Загрузка документов из %s", docs_path)
            documents = []
            for file in os.listdir(docs_path):
                file_path = os.path.join(docs_path, file)
                try:
                    if file.endswith(".txt"):
                        loader = TextLoader(file_path, encoding='utf-8')
                        documents.extend(loader.load())
                        logger.info("Загружен TXT: %s", file)
                    elif file.endswith(".pdf"):
                        loader = PyPDFLoader(file_path)
                        documents.extend(loader.load())
                        logger.info("Загружен PDF: %s", file)
                    else:
                        logger.warning("Пропущен файл: %s (не .txt/.pdf)", file)
                except Exception as e:
                    logger.error("Ошибка загрузки %s: %s", file, e)

            if not documents:
                logger.warning("Нет документов для индексации. RAG отключён.")
            else:
                # Лучший сплиттер для смешанных документов
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    separators=["\n\n", "\n", " ", ""]
                )
                docs = text_splitter.split_documents(documents)
                logger.info("Создано %d чанков", len(docs))

                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                )
                self.vectorstore = Chroma.from_documents(docs, embeddings)
                logger.info("Векторная БД создана")
        else:
            logger.warning("Папка %s пуста или не существует. RAG отключён.", docs_path)

        # 3. Память диалога
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=5
        )

    def get_response(self, user_message: str) -> str:
        if not self.vectorstore:
            return self.llm.invoke(user_message).content
        qa_chain = ConversationalRetrievalChain.from_llm(
            self.llm,
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 4}),
            memory=self.memory
        )
        result = qa_chain({"question": user_message})
        return result["answer"]