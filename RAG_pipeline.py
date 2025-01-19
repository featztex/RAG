from config import api_key
from utils import *

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA


def RAG_pipeline():
    """
    Создает и настраивает полный RAG пайплайн.
    
    Returns:
        RetrievalQA: Настроенная цепочка для вопросно-ответной системы
    """
    print("Начало работы")
    
    # Загрузка и подготовка данных
    texts = load_and_split_text()
    
    # Инициализация эмбеддингов
    print("Создание эмбеддингов...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sergeyzh/LaBSE-ru-sts",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Создание или загрузка векторного хранилища
    vectorstore = load_or_create_vectorstore(texts, embeddings)
    
    # Настройка retrievers
    ensemble_retriever = setup_retrievers(vectorstore, texts)
    
    # Инициализация языковой модели
    llm = initialize_llm(api_key)
    
    # Создание RAG цепочки
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=ensemble_retriever,
        return_source_documents=True
    )