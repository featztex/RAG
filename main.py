from config import api_key
from langchain_mistralai import ChatMistralAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

#import io, sys
#sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

mistral_api_key = api_key

# Загрузка данных и разделение на чанки
print("Загрузка данных...")
loader = TextLoader("data/content.txt")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, length_function=len)
texts = text_splitter.split_documents(documents)

# FAISS
print("Создание эмбеддингов...")
# sentence-transformers/all-mpnet-base-v2
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

print("Создание векторного хранилища...")
batch_size = 32
vectorstore = None

for i in tqdm(range(0, len(texts), batch_size), desc="Обработка батчей"):
    batch = texts[i:i + batch_size]
    if vectorstore is None:
        vectorstore = FAISS.from_documents(batch, embeddings)
    else:
        vectorstore.add_documents(batch)

print("Векторное хранилище создано")


# Инициализация модели и cоздание RAG-цепочки
llm = ChatMistralAI(mistral_api_key=mistral_api_key, model="mistral-tiny", timeout=30)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

def ask_rag(question):
    result = qa_chain({"query": question})
    return result["result"], result["source_documents"]

# Пример использования
question = "Умер ли Джон Сноу?"
answer, sources = ask_rag(question)

print(f"Ответ: {answer}\n")
print("Источники:")
for source in sources:
    print(f"- {source.page_content[:100]}...")