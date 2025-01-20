import telebot
from main import bot_mode
from config import bot_token

TOKEN = bot_token
bot = telebot.TeleBot(TOKEN)

@bot.message_handler(commands=['start'])
def start(message):
    bot.reply_to(message, 'Привет! Я готов отвечать на твои сообщения по сериалу "Игра престолов".')

@bot.message_handler(func=lambda message: True)
def handle_message(message):
    try:
        user_message = message.text
        response = bot_mode(user_message)
        bot.reply_to(message, response)
    except Exception as e:
        bot.reply_to(message, f"Произошла ошибка: {str(e)}")

if __name__ == "__main__":
    bot.polling()
