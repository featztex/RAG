from config import api_key
from langchain_mistralai import ChatMistralAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader

from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import os
import gc

mistral_api_key = api_key

def load_vectorstore(texts, embeddings, new=False):

    index_path = "faiss_index"
    vectorstore = None

    if os.path.exists(index_path) and new == False:
        print("Загрузка существующего индекса...")
        vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    else:
        print("Создание нового индекса...")
        vectorstore = create_vectorstore(texts, embeddings)
        vectorstore.save_local(index_path)
    
    return vectorstore

def create_vectorstore(texts, embeddings):

    print("Создание векторного хранилища...")
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
loader = TextLoader("data/content.txt")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

# FAISS
print("Создание эмбеддингов...")
# all-MiniLM-L6-v2, all-mpnet-base-v2
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vectorstore = load_vectorstore(texts, embeddings, new=True)


# Инициализация модели и cоздание RAG-цепочки
llm = ChatMistralAI(mistral_api_key=mistral_api_key, model="mistral-large-latest", timeout=30)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)


# Пример использования
def ask_rag(question):
    result = qa_chain.invoke({"query": question})
    return result["result"], result["source_documents"]

question = "Как звали разведчиков, отправившихся за стену в 1 серии?"
answer, sources = ask_rag(question)

print(f"Ответ: {answer}\n")
print("Источники:")
for source in sources:
    print(f"- {source.page_content[:100]}...")