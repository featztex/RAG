# RAG from scratch with Mistral API

## Описание

В репозитории реализован пайплайн **Retrieval-Augmented Generation (RAG)** для обработки запросов и генерации ответов на основе предоставленной базы знаний. Пайплайн реализует поиск релевантных документов и генерацию ответа с использованием MistralAI. 

В качестве данных используется лор сериала "Игра престолов". Данные были получены путем парсинга сайта [Game of Thrones Wiki](https://gameofthrones.fandom.com/ru). Был произведен парсинг содержания всех эпизодов сериала, а также отдельно биографий 50 основных героев.

## Установка

1. Клонируйте репозиторий
   ```bash
   git clone https://github.com/featztex/RAG.git
   cd RAG
   ```
2. Получите доступ к моделям MistralAI на [сайте](https://mistral.ai/) и сохраните его в файл config.py
    ```bash
   api_key = "<your MistralAI API key>"
   ```

## Пример использования
Пример запуска модели и комментарии к параметрам диалоговой функции можно найти в файле dialogue.py
```bash
from main import start_dialogue
start_dialogue()
```