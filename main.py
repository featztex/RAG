from config import api_key
from langchain_mistralai import ChatMistralAI
from RAG_pipeline import RAG_pipeline

import time
import gc
import re


# функции для формирования списка возможных ответов

def ask_rag(question, qa_chain):
    """
    Получает ответ на вопрос с помощью RAG системы.

    Args:
        question (str): Вопрос пользователя
        qa_chain: Цепочка вопрос-ответ RAG системы

    Returns:
        tuple: (ответ модели, использованные источники)
    """

    result = qa_chain.invoke({"query": question})
    return result["result"], result["source_documents"]


def get_paraphrased_queries(query, num_attempts, llm):
    """
    Генерирует перефразированные версии исходного запроса.

    Args:
        query (str): Исходный запрос
        num_attempts (int): Количество требуемых перефразировок
        llm: Языковая модель для перефразирования

    Returns:
        list: Список перефразированных запросов
    """
    if num_attempts <= 1:
        return [query]

    paraphrase_prompt = f"Перефразируй следующий вопрос {num_attempts} разными способами, \
                        сохраняя смысл. Напиши только перефразированные версии, \
                        каждую с новой строки: {query}"
    
    paraphrased = llm.invoke(paraphrase_prompt).content
    time.sleep(1)

    paraphrased_queries = [query] + [
        q.strip() for q in paraphrased.split('\n') 
        if q.strip()
    ][:num_attempts-1]
    
    return paraphrased_queries


def get_multiple_responses(query, qa_chain, num_attempts):
    """
    Получает несколько ответов на перефразированные версии запроса.

    Args:
        query (str): Исходный запрос
        qa_chain: Цепочка вопрос-ответ RAG системы
        num_attempts (int): Количество попыток получения ответа

    Returns:
        list: Список словарей с ответами и их метриками
    """

    llm = ChatMistralAI(
        mistral_api_key=api_key,
        model="mistral-large-latest",
        timeout=10
    )
    
    paraphrased_queries = get_paraphrased_queries(query, num_attempts, llm)
    responses = []

    for q in paraphrased_queries:
        answer, sources = ask_rag(q, qa_chain)
        responses.append({
            "query": q,
            "response": answer,
            "sources": sources,
            "confidence_score": calculate_confidence(answer, sources)
        })
        time.sleep(1)
        gc.collect()
    
    return responses


# функции для оценки качества ответа

def calculate_source_relevance(answer_lower, sources):
    """
    Вычисляет релевантность ответа относительно источников.

    Args:
        answer_lower (str): Ответ в нижнем регистре
        sources (list): Список источников

    Returns:
        float: Оценка релевантности
    """

    source_relevance = 0
    answer_words = set(answer_lower.split())

    for source in sources:
        source_text = source.page_content.lower()
        source_words = set(source_text.split())
        overlap = len(answer_words.intersection(source_words))
        source_relevance += overlap / len(answer_words) if answer_words else 0

    return source_relevance / len(sources) if sources else 0


def check_facts_presence(answer):
    """
    Проверяет наличие фактической информации в ответе.

    Args:
        answer (str): Ответ для проверки

    Returns:
        int: Количество найденных фактов
    """

    fact_indicators = [
        r'\d+',                        # Числа
        r'\d{1,2}:\d{2}',              # Время
        r'\d{1,2}\s+\w+\s+\d{4}',      # Даты
        r'[А-Я][а-я]+\s+[А-Я][а-я]+',  # Имена собственные
    ]

    facts_score = 0
    for pattern in fact_indicators:
        facts_count = len(re.findall(pattern, answer))
        facts_score += facts_count
    
    return facts_score


def calculate_confidence(answer, sources):
    """
    Вычисляет оценку уверенности для ответа на основе различных факторов.

    Args:
        answer (str): Ответ модели
        sources (list): Использованные источники

    Returns:
        float: Оценка уверенности
    """
    score = 0
    answer_lower = answer.lower()
    
    # Базовая оценка на основе длины
    score += len(answer.split()) * 0.05
    
    # Оценка согласованности с источниками
    score += calculate_source_relevance(answer_lower, sources) * 3.0

    # Бонус за факты
    score += check_facts_presence(answer)
    
    # Бонусы за структурированность
    structural_indicators = ["во-первых", "первое", "поскольку", "так как", "потому что", 
                            "суммируя", "подводя итог", "в итоге"]
    
    for indicator in structural_indicators:
        if indicator in answer_lower:
            score += 2
            break
    
    if ":" in answer:
        score += 1
    if len(answer.split('.')) > 2:
        score += 1
    
    # Корректировка за длину
    words_count = len(answer.split())
    if words_count < 8 or words_count > 100:
        score *= 0.8

    # Штрафы за незнание и/или неуверенность

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
        "недостаточно информации", "не могу точно сказать"
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
    Выбирает лучший ответ на основе оценки уверенности.

    Args:
        responses (list): Список ответов с их метриками

    Returns:
        tuple: (лучший ответ, источники лучшего ответа)
    """
    best_response = max(responses, key=lambda x: x["confidence_score"])
    return best_response["response"], best_response["sources"]


# функции ведения диалога с пользователем

def print_sources(sources, len_sources):
    """
    Выводит использованные источники.

    Args:
        sources (list): Список источников
        len_sources (int): Длина выводимого фрагмента источника
    """
    print('Источники:')
    for s in sources:
        print(f"- ...{s.page_content[:len_sources]}...\n")


def print_all_responses(responses, best_answer):
    """
    Выводит все полученные ответы и их метрики.

    Args:
        responses (list): Список ответов с их метриками
        best_answer (str): Лучший ответ
    """
    print("\n=== Все варианты ответов ===")
    for i, resp in enumerate(responses, 1):
        print(f"\n--- Вариант {i} ---")
        print(f"Перефразированный вопрос: {resp['query']}")
        print(f"Ответ: {resp['response']}")
        print(f"Оценка уверенности: {resp['confidence_score']:.2f}")
    print(f"\nЛучший ответ: {best_answer}\n")


def start_dialogue(sources=False, len_sources=None, num_attempts=1, all_answers=False):
    """
    Запускает диалоговый интерфейс RAG системы.

    Args:
        sources (bool): Флаг вывода источников
        len_sources (int): Длина выводимого фрагмента источника
        num_attempts (int): Количество попыток получения ответа
        all_answers (bool): Флаг вывода всех полученных ответов
    """

    qa_chain = RAG_pipeline()
    print(f"\nRAG-система готова. Введите Ваш вопрос. \
          \nДля выхода введите 'выход'\n")

    while True:
        user_input = input("Ваш вопрос: ")
        if user_input.lower() == 'выход':
            break
        
        responses = get_multiple_responses(user_input, qa_chain, num_attempts)
        best_answer, best_sources = select_best_response(responses)

        if all_answers:
            print_all_responses(responses, best_answer)
        else:
            print(f"Ответ: {best_answer}\n")
        
        if sources:
            print_sources(best_sources, len_sources)