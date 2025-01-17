from config import api_key
from langchain_mistralai import ChatMistralAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader

#import io, sys
#sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

mistral_api_key = api_key

# Загрузка данных и разделение на чанки
loader = TextLoader("data/content.txt")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, length_function=len)
texts = text_splitter.split_documents(documents)

# FAISS
print("Создание эмбеддингов...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
print("Создание векторного хранилища...")
vectorstore = FAISS.from_documents(texts, embeddings)
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