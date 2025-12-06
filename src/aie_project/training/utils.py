import dotenv


def find_and_load_dotenv():
    if dotenv.find_dotenv():
        dotenv.load_dotenv(dotenv.find_dotenv())
