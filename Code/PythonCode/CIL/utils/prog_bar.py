from tqdm.contrib import telegram
from tqdm import trange
import os

from setting import BOT_API_TOKEN, CHAT_ROOM_ID


def prog_bar(epochs):
    TELEGRAM_USING = os.getenv("TELEGRAM_USING", default=0)
    if int(TELEGRAM_USING) == 0:
        return trange(epochs)
    return telegram.trange(epochs, token=BOT_API_TOKEN, chat_id=CHAT_ROOM_ID, mininterval=3)
