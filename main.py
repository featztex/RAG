from config import api_key
from langchain_mistralai import ChatMistralAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain_community.retrievers import TFIDFRetriever
from langchain.retrievers import EnsembleRetriever

from tqdm import tqdm
import time
import os
import gc
import re

def get_multiple_responses(query, qa_chain, num_attempts=5):
    """
    Получает несколько ответов на перефразированные версии запроса
    """
    responses = []
    
    # Получаем перефразированные версии запроса
    paraphrase_prompt = f"Перефразируй следующий вопрос {num_attempts} разными способами, сохраняя смысл. Напиши только перефразированные версии, каждую с новой строки: {query}"
    
    llm = ChatMistralAI(
        mistral_api_key=api_key,
        model="mistral-large-latest",
        timeout=10
    )
    
    # Исправляем извлечение текста из ответа модели
    paraphrased = llm.invoke(paraphrase_prompt).content
    paraphrased_queries = [query] + [q.strip() for q in paraphrased.split('\n') if q.strip()][:num_attempts-1]
    time.sleep(1)

    # Получаем ответы на все версии запроса
    for q in paraphrased_queries:
        answer, sources = ask_rag(q, qa_chain)
        responses.append({
            "query": q,
            "response": answer,
            "sources": sources,
            "confidence_score": calculate_confidence(answer, sources)
        })
        time.sleep(1)
    
    return responses


def calculate_confidence(answer, sources):
    """
    Вычисляет оценку уверенности для ответа на основе различных факторов
    """
    score = 0
    answer_lower = answer.lower()
    
    # Длина ответа 
    score += len(answer.split()) * 0.05
    
    # Оценка согласованности с источниками
    source_relevance = 0

    for source in sources:
        source_text = source.page_content.lower()
        answer_words = set(answer_lower.split())
        source_words = set(source_text.split())
        overlap = len(answer_words.intersection(source_words))
        source_relevance += overlap / len(answer_words) if answer_words else 0

    source_relevance = source_relevance / len(sources) if sources else 0
    score += source_relevance * 3.0
    
    # Бонусы за структурированность ответа
    if "во-первых" in answer_lower or "первое" or "поскольку" or "так как" in answer_lower:
        score += 2
    if ":" in answer:
        score += 1
    if len(answer.split('.')) > 2:
        score += 1
    
    # Бонус за конкретные факты
    fact_indicators = [
        r'\d+',                         # Числа
        r'\d{1,2}:\d{2}',               # Время
        r'\d{1,2}\s+\w+\s+\d{4}',       # Даты
        r'[А-Я][а-я]+\s+[А-Я][а-я]+',   # Имена собственные
    ]
    
    for pattern in fact_indicators:
        facts_count = len(re.findall(pattern, answer))
        score += facts_count
        
    # Штрафы за слишком короткие или длинные ответы
    words_count = len(answer.split())
    if words_count < 8:
        score *= 0.8
    elif words_count > 100:
        score *= 0.8

    # Неуверенные ответы
    uncertainty_phrases = [
        "возможно", "вероятно", "предположительно",
        "может быть", "вроде", "вроде бы",
        "вроде как", "кажется", "кажись",
        "будто", "как будто", "наверно",
        "наверное", "видимо", "по-видимому",
        "похоже", "должно быть",
        "трудно сказать", "сложно утверждать",
        "я не уверен", "это спорный вопрос",
        "нельзя сказать точно", "под вопросом",
        "требует уточнения", "нет точных данных",
        "недостаточно информации", "не могу точно сказать",
        "примерно", "около", "приблизительно",
        "где-то", "как-то так", "типа того",
        "в некотором роде", "своего рода",
        "отчасти", "в какой-то степени",
        "более-менее", "относительно"
    ]
    
    # Прямые указания на незнание
    ignorance_phrases = [
        "не знаю", "не указан", "не могу сказать",
        "точно не известно", "нет информации", "нет данных",
        "не совсем ясно", "затрудняюсь ответить",
        "информация отсутствует", "не располагаю информацией",
    ]
    
    for phrase in uncertainty_phrases:
        if phrase in answer_lower:
            score *= 0.7
    for phrase in ignorance_phrases:
        if phrase in answer_lower:
            score *= 0.2
    
    return score

def select_best_response(responses):
    """
    Выбирает лучший ответ на основе оценки уверенности
    """
    best_response = max(responses, key=lambda x: x["confidence_score"])
    return best_response["response"], best_response["sources"]



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

    # Создание векторного хранилища FAISS
    print("Создание эмбеддингов...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sergeyzh/LaBSE-ru-sts",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    vectorstore = load_vectorstore(texts, embeddings, new=False)

     # Создание ретривера
    faiss_retriever = vectorstore.as_retriever(
        search_type="similarity",  # mmr (lambda_mult), similarity_score_threshold (score_threshold)
        search_kwargs={'k': 5, 'fetch_k': 15}
    )

    tfidf_retriever = TFIDFRetriever.from_documents(texts, k=5)

    ensemble_retriever = EnsembleRetriever(
        retrievers=[faiss_retriever, tfidf_retriever],
        weights=[0.75, 0.25]
    )

    # Инициализация модели и cоздание RAG-цепочки
    llm = ChatMistralAI(
        mistral_api_key=mistral_api_key, 
        model="mistral-large-latest", 
        timeout=10
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=ensemble_retriever,
        return_source_documents=True
    )

    return qa_chain



def ask_rag(question, qa_chain):
        result = qa_chain.invoke({"query": question})
        return result["result"], result["source_documents"]



def start_dialogue(sources=False, len_sources=100):
    qa_chain = RAG_pipline()
    print("\nRAG-система готова. Введите Ваш вопрос. \
          \nСистема автоматически перефразирует ваш вопрос и выберет лучший ответ. \
          \nДля выхода введите 'выход'\n")

    while True:
        user_input = input("Ваш вопрос: ")
        if user_input.lower() == 'выход':
            break
        
        print("Генерация нескольких вариантов ответа...")
        responses = get_multiple_responses(user_input, qa_chain)
        best_answer, best_sources = select_best_response(responses)

        print("\n=== Все варианты ответов ===")
        for i, resp in enumerate(responses, 1):
            print(f"\n--- Вариант {i} ---")
            print(f"Перефразированный вопрос: {resp['query']}")
            print(f"Ответ: {resp['response']}")
            print(f"Оценка уверенности: {resp['confidence_score']:.2f}")
        
        print(f"\nЛучший ответ: {best_answer}\n")
        
        if sources:
            print('Источники:')
            for s in best_sources:
                print(f"- ...{s.page_content[:len_sources]}...\n")