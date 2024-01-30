import telebot
BOT_API_TOKEN = "6127832852:AAFbDMjHN8zed9uZjVodmWn4hO3sdS6Pi-U"
CHAT_ROOM_ID = "-4083757988"


def send_telegram_notification():
    bot = telebot.TeleBot(BOT_API_TOKEN)
    bot.send_message(chat_id=CHAT_ROOM_ID, text="Finish training model")
