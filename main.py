from config import api_key
from langchain_mistralai import ChatMistralAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader

from tqdm import tqdm
import os
import gc

def RAG_pipline():

    mistral_api_key = api_key

    def split_text(chunk_size=500, chunk_overlap=50, data_path="data/all_content.txt"):

        loader = TextLoader(data_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        texts = text_splitter.split_documents(documents)

        return texts

    def load_vectorstore(texts, embeddings, new=False):

        index_path = "faiss_index"
        vectorstore = None
        print("Загрузка векторного хранилища...")

        if os.path.exists(index_path) and new == False:
            vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        else:
            vectorstore = create_vectorstore(texts, embeddings)
            vectorstore.save_local(index_path)
        
        return vectorstore

    def create_vectorstore(texts, embeddings):

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


    # Загрузка данных и разделение на чанки
    print("Начало работы")
    texts = split_text(chunk_size=500, chunk_overlap=50)

    # Создание векторного хранилища
    print("Создание эмбеддингов...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sergeyzh/LaBSE-ru-sts",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vectorstore = load_vectorstore(texts, embeddings, new=False)


    # Инициализация модели и cоздание RAG-цепочки
    llm = ChatMistralAI(
        mistral_api_key=mistral_api_key, 
        model="mistral-large-latest", 
        timeout=30
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_type="similarity", # mmr (lambda_mult), similarity_score_threshold (score_threshold)
            search_kwargs={'k': 8, 'fetch_k': 20}
        ),
        return_source_documents=True
    )

    return qa_chain

def ask_rag(question, qa_chain):
        result = qa_chain.invoke({"query": question})
        return result["result"], result["source_documents"]

#RAG_pipline()