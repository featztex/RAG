from config import api_key
from langchain_mistralai import ChatMistralAI
from RAG_pipeline import RAG_pipeline

import time
import gc
import re


def ask_rag(question, qa_chain):
        result = qa_chain.invoke({"query": question})
        return result["result"], result["source_documents"]


def get_multiple_responses(query, qa_chain, num_attempts):
    """
    Получает несколько ответов на перефразированные версии запроса
    """
    responses = []
    
    # Получаем перефразированные версии запроса
    if num_attempts > 1:
    
        paraphrase_prompt = f"Перефразируй следующий вопрос {num_attempts} разными способами, сохраняя смысл. Напиши только перефразированные версии, каждую с новой строки: {query}"
        
        llm = ChatMistralAI(
            mistral_api_key=api_key,
            model="mistral-large-latest",
            timeout=10
        )
        
        paraphrased = llm.invoke(paraphrase_prompt).content
        paraphrased_queries = [query] + [q.strip() for q in paraphrased.split('\n') if q.strip()][:num_attempts-1]
        time.sleep(1)
    else:
        paraphrased_queries = [query]

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
        gc.collect()
    
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
    Выбирает лучший ответ на основе оценки уверенности
    """
    best_response = max(responses, key=lambda x: x["confidence_score"])
    return best_response["response"], best_response["sources"]



def start_dialogue(sources=False, len_sources=None, num_attempts=1, all_answers=False):
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
            print("\n=== Все варианты ответов ===")
            for i, resp in enumerate(responses, 1):
                print(f"\n--- Вариант {i} ---")
                print(f"Перефразированный вопрос: {resp['query']}")
                print(f"Ответ: {resp['response']}")
                print(f"Оценка уверенности: {resp['confidence_score']:.2f}")
            print(f"\nЛучший ответ: {best_answer}\n")
        else:
            print(f"Ответ: {best_answer}\n")
        
        if sources:
            print('Источники:')
            for s in best_sources:
                print(f"- ...{s.page_content[:len_sources]}...\n")