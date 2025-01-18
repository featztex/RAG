from main import RAG_pipline, ask_rag

qa_chain = RAG_pipline()

print("RAG-система готова. Введите вопрос или 'выход' для завершения.")
while True:
    user_input = input("Ваш вопрос: ")
    if user_input.lower() == 'выход':
        break
    answer, sources = ask_rag(user_input, qa_chain)
    print(f"Ответ: {answer}\n")
    print('Источники:')
    for source in sources:
        print(f"- ...{source.page_content[:200]}...\n")