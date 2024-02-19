from setting import BOT_API_TOKEN, CHAT_ROOM_ID
import telebot


def send_telegram_notification(text: str):
    try:
        bot = telebot.TeleBot(BOT_API_TOKEN)
        bot.send_message(chat_id=CHAT_ROOM_ID, text=text)
    except telebot.apihelper.ApiTelegramException:
        print("Error when sending telegram message")
