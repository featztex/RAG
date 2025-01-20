import telebot
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton
from main import get_multiple_responses, select_best_response
from RAG_pipeline import RAG_pipeline
from config import bot_token

TOKEN = bot_token
bot = telebot.TeleBot(TOKEN)

user_settings = {}
setup_messages = {}

def create_yes_no_keyboard():
    keyboard = InlineKeyboardMarkup()
    keyboard.row(InlineKeyboardButton("Да", callback_data="yes"),
                 InlineKeyboardButton("Нет", callback_data="no"))
    return keyboard

def show_final_settings(chat_id, user_id):
    settings = user_settings[user_id]
    summary = "📋 Ваши настройки:\n\n"
    
    summary += "🔍 Показывать источники: "
    summary += "Да" if settings['show_sources'] else "Нет"
    
    if settings['show_sources']:
        summary += f"\n📏 Длина фрагмента источников: {settings['source_length']} символов"
    
    summary += f"\n🔄 Количество попыток вызова LLM: {settings['num_attempts']}"
    
    summary += "\n📝 Формат вывода: "
    summary += "Все ответы" if settings['all_answers'] else "Только лучший ответ"
    
    summary += "\n\n✅ Настройка завершена. Можете задавать вопросы!"
    
    bot.send_message(chat_id, summary)


@bot.message_handler(commands=['start'])
def start(message):
    user_id = message.from_user.id
    setup_messages[user_id] = []
    init_msg = bot.reply_to(message, 'Инициализация RAG-системы...')
    setup_messages[user_id].append(init_msg.message_id)
    
    user_settings[user_id] = {'qa_chain': RAG_pipeline()}
    bot.reply_to(message, 'Привет! Я готов отвечать на твои вопросы по сериалу "Игра престолов". Для начала нужно провести настройку бота.')
    ask_sources(message)


def ask_sources(message):
    msg = bot.send_message(message.chat.id, "Выводить ли список фрагментов лора, используемых для формирования ответа?", reply_markup=create_yes_no_keyboard())
    setup_messages[message.from_user.id].append(msg.message_id)


@bot.callback_query_handler(func=lambda call: call.data in ["yes", "no"])
def callback_sources(call):
    user_id = call.from_user.id
    if user_id not in user_settings:
        user_settings[user_id] = {}
    if user_id not in setup_messages:
        setup_messages[user_id] = []
        
    user_settings[user_id]['show_sources'] = call.data == "yes"
    bot.answer_callback_query(call.id)
    
    if call.data == "yes":
        msg = bot.send_message(call.message.chat.id, "Введите длину фрагмента источников (число символов):")
        setup_messages[user_id].append(msg.message_id)
        bot.register_next_step_handler(call.message, set_source_length)
    else:
        ask_attempts(call.message)

def ask_attempts(message):
    user_id = message.from_user.id
    if user_id not in setup_messages:
        setup_messages[user_id] = []
        
    msg = bot.send_message(message.chat.id, "Сколько попыток обращений к LLM использовать? Введите число от 1 до 6. Большее значение повышает качество ответа, но заметно увеличивает время его ожидания.")
    setup_messages[user_id].append(msg.message_id)
    bot.register_next_step_handler(message, set_attempts)

def ask_attempts(message):
    user_id = message.from_user.id
    if user_id not in setup_messages:
        setup_messages[user_id] = []
        
    msg = bot.send_message(message.chat.id, "Сколько попыток обращений к LLM использовать? Введите число от 1 до 6. Большее значение повышает качество ответа, но заметно увеличивает время его ожидания.")
    setup_messages[user_id].append(msg.message_id)
    setup_messages[user_id].append(message.message_id)  # Добавляем ID исходного сообщения
    bot.register_next_step_handler(message, set_attempts)

def set_attempts(message):
    user_id = message.from_user.id
    if user_id not in setup_messages:
        setup_messages[user_id] = []
    
    setup_messages[user_id].append(message.message_id)  # Добавляем ID ответа пользователя
        
    try:
        attempts = int(message.text)
        if 1 <= attempts <= 6:
            user_settings[user_id]['num_attempts'] = attempts
            ask_all_answers(message)
        else:
            msg = bot.reply_to(message, "Пожалуйста, введите число от 1 до 6.")
            setup_messages[user_id].append(msg.message_id)
            bot.register_next_step_handler(message, set_attempts)
    except ValueError:
        msg = bot.reply_to(message, "Пожалуйста, введите число.")
        setup_messages[user_id].append(msg.message_id)
        bot.register_next_step_handler(message, set_attempts)



def set_source_length(message):
    user_id = message.from_user.id
    setup_messages[user_id].append(message.message_id)
    try:
        length = int(message.text)
        user_settings[user_id]['source_length'] = length
        ask_attempts(message)
    except ValueError:
        msg = bot.reply_to(message, "Пожалуйста, введите число.")
        setup_messages[user_id].append(msg.message_id)
        bot.register_next_step_handler(message, set_source_length)


def ask_all_answers(message):
    keyboard = InlineKeyboardMarkup()
    keyboard.row(InlineKeyboardButton("Лучший ответ", callback_data="best"),
                 InlineKeyboardButton("Все ответы", callback_data="all"))
    msg = bot.send_message(message.chat.id, "Выводить все полученные ответы или только лучший?", reply_markup=keyboard)
    setup_messages[message.from_user.id].append(msg.message_id)

@bot.callback_query_handler(func=lambda call: call.data in ["best", "all"])
def callback_answers(call):
    user_id = call.from_user.id
    user_settings[user_id]['all_answers'] = call.data == "all"
    bot.answer_callback_query(call.id)
    
    # Удаляем все сообщения настройки
    for msg_id in setup_messages[user_id]:
        try:
            bot.delete_message(call.message.chat.id, msg_id)
        except Exception:
            pass
    
    setup_messages[user_id] = []
    
    # Показываем итоговые настройки
    show_final_settings(call.message.chat.id, user_id)


@bot.message_handler(commands=['end'])
def end_dialogue(message):
    bot.reply_to(message, "Диалог завершен. Бот остановлен.")
    bot.stop_polling()

@bot.message_handler(func=lambda message: True)
def handle_message(message):
    user_id = message.from_user.id
    if user_id not in user_settings:
        bot.reply_to(message, "Пожалуйста, начните диалог с команды /start")
        return

    try:
        user_message = message.text
        qa_chain = user_settings[user_id]['qa_chain']
        num_attempts = user_settings[user_id]['num_attempts']
        
        responses = get_multiple_responses(user_message, qa_chain, num_attempts)
        best_answer, best_sources = select_best_response(responses)

        if user_settings[user_id]['all_answers']:
            response = "=== Все варианты ответов ===\n\n"
            for i, resp in enumerate(responses, 1):
                response += f"--- Вариант {i} ---\n"
                response += f"Перефразированный вопрос: {resp['query']}\n"
                response += f"Ответ: {resp['response']}\n"
                response += f"Оценка уверенности: {resp['confidence_score']:.2f}\n\n"
            response += f"Лучший ответ: {best_answer}\n"
        else:
            response = f"Ответ: {best_answer}\n"

        if user_settings[user_id]['show_sources']:
            response += "\nИсточники:\n"
            for s in best_sources:
                response += f"- ...{s.page_content[:user_settings[user_id]['source_length']]}...\n"

        bot.reply_to(message, response)
    except Exception as e:
        bot.reply_to(message, f"Произошла ошибка: {str(e)}")

if __name__ == "__main__":
    bot.polling()