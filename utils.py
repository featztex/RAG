from config import api_key
from langchain_mistralai import ChatMistralAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_community.retrievers import TFIDFRetriever
from langchain.retrievers import EnsembleRetriever

from tqdm import tqdm
import time
import os
import gc


def load_and_split_text(data_path="data/all_content.txt", chunk_size=500, chunk_overlap=50):
    """
    Загружает текстовые данные из файла и разделяет их на чанки.
    
    Args:
        data_path (str): Путь к текстовому файлу
        chunk_size (int): Размер каждого чанка текста
        chunk_overlap (int): Размер перекрытия между чанками
    
    Returns:
        list: Список документов, разделенных на чанки
    """

    loader = TextLoader(data_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)

    return texts


def create_vectorstore(texts, embeddings):
    """
    Создает новое векторное хранилище из текстовых документов.
    
    Args:
        texts (list): Список документов для индексации
        embeddings: Модель для создания эмбеддингов
        batch_size (int): Размер батча для обработки документов
    
    Returns:
        FAISS: Векторное хранилище
    """

    print("Создание нового векторного хранилища...")
    batch_size = 16
    vectorstore = None

    for i in tqdm(range(0, len(texts), batch_size), desc="Обработка батчей"):
        batch = texts[i:i + batch_size]
        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embeddings)
        else:
            vectorstore.add_documents(batch)
        gc.collect()

    print("Векторное хранилище создано")
    return vectorstore


def load_or_create_vectorstore(texts, embeddings, new=False):
    """
    Загружает существующее векторное хранилище или создает новое.
    
    Args:
        texts (list): Список документов для индексации
        embeddings: Модель для создания эмбеддингов
        index_path (str): Путь к файлу индекса
        new (bool): Флаг для создания нового хранилища
    
    Returns:
        FAISS: Векторное хранилище
    """

    index_path = "faiss_index"
    vectorstore = None

    if os.path.exists(index_path) and new == False:
        vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    else:
        vectorstore = create_vectorstore(texts, embeddings)
        vectorstore.save_local(index_path)
    
    return vectorstore


def setup_retrievers(vectorstore, texts):
    """
    Настраивает и комбинирует FAISS & TF-IDF retrievers.
    
    Args:
        vectorstore (FAISS): Векторное хранилище FAISS
        texts (list): Список документов для TF-IDF
    
    Returns:
        EnsembleRetriever: Комбинированный retriever
    """

    faiss_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={'k': 5, 'fetch_k': 15}
    )

    tfidf_retriever = TFIDFRetriever.from_documents(texts, k=5)

    return EnsembleRetriever(
        retrievers=[faiss_retriever, tfidf_retriever],
        weights=[0.75, 0.25]
    )


def initialize_llm(api_key, model_name="mistral-large-latest"):
    """
    Инициализирует языковую модель Mistral.
    
    Args:
        api_key (str): API ключ для Mistral
        model_name (str): название испльзуемой модели
    
    Returns:
        ChatMistralAI: Инициализированная модель
    """
    return ChatMistralAI(
        mistral_api_key=api_key,
        model=model_name,
        timeout=10
    )