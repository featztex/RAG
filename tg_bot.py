import telebot
from telebot.types import ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardMarkup, InlineKeyboardButton
from main import ask_rag, get_multiple_responses, select_best_response, print_sources, print_all_responses
from RAG_pipeline import RAG_pipeline
from config import bot_token

TOKEN = bot_token
bot = telebot.TeleBot(TOKEN)

user_settings = {}

def create_yes_no_keyboard():
    keyboard = InlineKeyboardMarkup()
    keyboard.row(InlineKeyboardButton("Да", callback_data="yes"),
                 InlineKeyboardButton("Нет", callback_data="no"))
    return keyboard

@bot.message_handler(commands=['start'])
def start(message):
    bot.reply_to(message, 'Инициализация RAG-системы...')
    user_id = message.from_user.id
    user_settings[user_id] = {'qa_chain': RAG_pipeline()}
    bot.reply_to(message, 'Привет! Я готов отвечать на твои вопросы по сериалу "Игра престолов".')
    ask_sources(message)



def ask_sources(message):
    bot.send_message(message.chat.id, "Выводить ли список используемых для ответа фрагментов лора?", reply_markup=create_yes_no_keyboard())

@bot.callback_query_handler(func=lambda call: call.data in ["yes", "no"])
def callback_sources(call):
    user_id = call.from_user.id
    user_settings[user_id]['show_sources'] = call.data == "yes"
    if call.data == "yes":
        # Удаляем сообщение с кнопками
        bot.delete_message(call.message.chat.id, call.message.message_id)
        # Отправляем новое сообщение
        bot.send_message(call.message.chat.id, "Введите длину фрагмента источников (число символов):")
        bot.register_next_step_handler(call.message, set_source_length)
    else:
        ask_attempts(call.message)

def set_source_length(message):
    user_id = message.from_user.id
    try:
        length = int(message.text)
        user_settings[user_id]['source_length'] = length
        ask_attempts(message)
    except ValueError:
        bot.reply_to(message, "Пожалуйста, введите число.")
        bot.register_next_step_handler(message, set_source_length)


def ask_attempts(message):
    bot.send_message(message.chat.id, "Сколько попыток обращений к LLM использовать? (введите число от 1 до 6)")
    bot.register_next_step_handler(message, set_attempts)

def set_attempts(message):
    user_id = message.from_user.id
    try:
        attempts = int(message.text)
        if 1 <= attempts <= 6:
            user_settings[user_id]['num_attempts'] = attempts
            ask_all_answers(message)
        else:
            bot.reply_to(message, "Пожалуйста, введите число от 1 до 6.")
            bot.register_next_step_handler(message, set_attempts)
    except ValueError:
        bot.reply_to(message, "Пожалуйста, введите число.")
        bot.register_next_step_handler(message, set_attempts)

def ask_all_answers(message):
    keyboard = InlineKeyboardMarkup()
    keyboard.row(InlineKeyboardButton("Лучший ответ", callback_data="best"),
                 InlineKeyboardButton("Все ответы", callback_data="all"))
    bot.send_message(message.chat.id, "Выводить все полученные ответы или только лучший?", reply_markup=keyboard)

@bot.callback_query_handler(func=lambda call: call.data in ["best", "all"])
def callback_answers(call):
    user_id = call.from_user.id
    user_settings[user_id]['all_answers'] = call.data == "all"
    bot.send_message(call.message.chat.id, "Настройка завершена. Задайте свой вопрос!")



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