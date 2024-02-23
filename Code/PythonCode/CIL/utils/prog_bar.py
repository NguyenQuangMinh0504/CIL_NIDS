from tqdm.contrib.telegram import trange
from setting import BOT_API_TOKEN, CHAT_ROOM_ID


def prog_bar(epochs):
    return trange(epochs, token=BOT_API_TOKEN, chat_id=CHAT_ROOM_ID, mininterval=3)
